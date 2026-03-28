import torch
import numpy as np
import trimesh
import nvdiffrast.torch as dr

class NvdiffrastRenderer:
    def __init__(self, mesh_path: str, device: str = "cuda"):
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device=self.device)

        # 1. Mesh 로드 (GLB Scene 구조 안전 확보 및 빈 씬 방어)
        obj = trimesh.load(mesh_path, process=False)
        if isinstance(obj, trimesh.Scene):
            meshes = [g for g in obj.dump(concatenate=False) if isinstance(g, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError(f"No mesh geometries found in scene: {mesh_path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = obj
        
        # 2. Vertex & Face 텐서화
        self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(mesh.faces, dtype=torch.int32, device=self.device).contiguous()

        # 3. 색상 로드 및 Topology Hash 캐싱 (속도 병목 제거)
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            colors = np.ones((len(mesh.vertices), 3)) * 0.8
        self.colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        
        # Antialias 계산을 위한 위상 정보 사전 생성 (반복 연산 절약)
        self.topology_hash = dr.antialias_construct_topology_hash(self.faces)

    def _get_projection_matrix(self, K, H: int, W: int, znear: float = 0.01, zfar: float = 100.0) -> torch.Tensor:
        proj = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj[0, 0] = 2.0 * K.fx / W
        proj[0, 2] = 2.0 * K.cx / W - 1.0
        proj[1, 1] = -2.0 * K.fy / H
        proj[1, 2] = 1.0 - 2.0 * K.cy / H
        proj[2, 2] = (zfar + znear) / (zfar - znear)
        proj[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
        proj[3, 2] = 1.0
        return proj

    def render_batch(self, A_batch: torch.Tensor, b_batch: torch.Tensor, K) -> dict:
        """
        다수의 포즈를 한 번에 렌더링 (Seed Search 병목 해결용)
        - A_batch: [B, 3, 3]
        - b_batch: [B, 3]
        """
        B = A_batch.shape[0]
        H, W = int(K.height), int(K.width)

        # 1. 아핀 변환
        V = self.vertices.unsqueeze(0).expand(B, -1, -1)
        AT = A_batch.transpose(1, 2).contiguous()

        x = V[..., 0]
        y = V[..., 1]
        z = V[..., 2]

        vx = x * AT[:, None, 0, 0] + y * AT[:, None, 1, 0] + z * AT[:, None, 2, 0]
        vy = x * AT[:, None, 0, 1] + y * AT[:, None, 1, 1] + z * AT[:, None, 2, 1]
        vz = x * AT[:, None, 0, 2] + y * AT[:, None, 1, 2] + z * AT[:, None, 2, 2]

        v_trans = torch.stack([vx, vy, vz], dim=-1) + b_batch.unsqueeze(1)


        # 2. 투영
        proj = self._get_projection_matrix(K, H, W)
        v_homo = torch.cat([v_trans, torch.ones_like(v_trans[..., :1])], dim=-1)

        proj_t = proj.T.contiguous()
        v_clip = torch.stack([
            v_homo[..., 0] * proj_t[0, 0] + v_homo[..., 1] * proj_t[1, 0] + v_homo[..., 2] * proj_t[2, 0] + v_homo[..., 3] * proj_t[3, 0],
            v_homo[..., 0] * proj_t[0, 1] + v_homo[..., 1] * proj_t[1, 1] + v_homo[..., 2] * proj_t[2, 1] + v_homo[..., 3] * proj_t[3, 1],
            v_homo[..., 0] * proj_t[0, 2] + v_homo[..., 1] * proj_t[1, 2] + v_homo[..., 2] * proj_t[2, 2] + v_homo[..., 3] * proj_t[3, 2],
            v_homo[..., 0] * proj_t[0, 3] + v_homo[..., 1] * proj_t[1, 3] + v_homo[..., 2] * proj_t[2, 3] + v_homo[..., 3] * proj_t[3, 3],
        ], dim=-1)

        # 3. Rasterization
        pos = v_clip.contiguous()
        rast, _ = dr.rasterize(self.glctx, pos, self.faces, resolution=[H, W])

        # 4. Soft Alpha (가장 중요한 미분 포인트 + 값 범위 안전성 확보)
        hard_mask = (rast[..., 3:] > 0).float()
        soft_alpha = dr.antialias(
            hard_mask, rast, pos, self.faces, topology_hash=self.topology_hash
        ).clamp(0.0, 1.0)

        # 5. 보간
        depth, _ = dr.interpolate(v_trans[..., 2:3].contiguous(), rast, self.faces)
        depth = depth * hard_mask

        C = self.colors.unsqueeze(0).expand(B, -1, -1).contiguous()
        color, _ = dr.interpolate(C, rast, self.faces)
        color = color * hard_mask

        # 6. OpenCV 좌표계 호환을 위한 Y축 반전 (OpenGL Bottom-up 이슈 해결)
        hard_mask = torch.flip(hard_mask, dims=[1])
        soft_alpha = torch.flip(soft_alpha, dims=[1])
        depth = torch.flip(depth, dims=[1])
        color = torch.flip(color, dims=[1])

        return {
            'mask': hard_mask.squeeze(-1),   # [B, H, W] - Debug용 Hard
            'alpha': soft_alpha.squeeze(-1), # [B, H, W] - 최적화용 Soft
            'depth': depth.squeeze(-1),      # [B, H, W]
            'rgb': color                     # [B, H, W, 3]
        }

    def render_torch(self, A_3x3: torch.Tensor, b_xyz: torch.Tensor, K) -> dict:
        """ 단일 포즈 미분 렌더링 래퍼 """
        res = self.render_batch(A_3x3.unsqueeze(0), b_xyz.unsqueeze(0), K)
        return {k: v[0] for k, v in res.items()}

    def render(self, A_3x3: np.ndarray, b_xyz: np.ndarray, K) -> dict:
        """ 기존 평가용 NumPy 래퍼 """
        A_t = torch.tensor(A_3x3, dtype=torch.float32, device=self.device)
        b_t = torch.tensor(b_xyz, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            res = self.render_torch(A_t, b_t, K)
            
            mask_np = res['mask'].cpu().numpy().astype(bool)
            depth_np = res['depth'].cpu().numpy()
            
            rgb_np = res['rgb'].cpu().numpy()
            rgb_np = np.clip(rgb_np, 0.0, 1.0)
            rgb_np = (rgb_np * 255).astype(np.uint8)
            bgr_np = rgb_np[..., ::-1].copy() if rgb_np.shape[-1] == 3 else rgb_np
            alpha_np = res['alpha'].cpu().numpy().astype(np.float32)
            alpha_np = np.clip(alpha_np, 0.0, 1.0)



        return {
            'mask': mask_np,
            'depth': depth_np,
            'rgb': bgr_np,
            'alpha': alpha_np,
        }