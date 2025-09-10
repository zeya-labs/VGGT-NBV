"""
网格生成模块
处理网格的生成、保存和PyTorch3D mesh创建
"""

import os
import numpy as np
import torch
import noise
from PIL import Image
from typing import Dict, Any
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesUV

from .coordinate_utils import get_coordinate_transform_matrix, apply_transform_to_vertices


class MeshGenerator:
    """网格生成器类"""
    
    def __init__(self, up_axis: str = "Y"):
        """
        初始化网格生成器
        
        Args:
            up_axis: 上朝向轴
        """
        self.up_axis = up_axis
        self.transform_matrix = get_coordinate_transform_matrix(up_axis)
    
    def generate_textured_perlin_sphere(
        self, 
        seed: int, 
        scale: float = 0.5, 
        octaves: int = 6, 
        persistence: float = 0.5, 
        lacunarity: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        生成带有Perlin噪声纹理的球体网格
        
        Args:
            seed: 随机种子
            scale: 噪声缩放
            octaves: 噪声八度
            persistence: 噪声持续性
            lacunarity: 噪声间隙
            
        Returns:
            mesh_data: 包含vertices, faces, uvs的字典
        """
        np.random.seed(seed)

        num_phi, num_theta = 40, 80
        phi_vals = np.linspace(0, np.pi, num_phi)
        theta_vals = np.linspace(0, 2 * np.pi, num_theta)

        vertices = []
        uvs = []

        for i in range(num_phi):
            for j in range(num_theta):
                phi, theta = phi_vals[i], theta_vals[j]
                x_base = np.sin(phi) * np.cos(theta)
                y_base = np.cos(phi)
                z_base = np.sin(phi) * np.sin(theta)

                noise_val = noise.pnoise3(
                    x_base * scale,
                    y_base * scale,
                    z_base * scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    repeatz=1024,
                    base=seed,
                )
                radius = 1.0 + 0.3 * noise_val

                vertices.append([radius * x_base, radius * y_base, radius * z_base])

                u = 1.0 - j / (num_theta - 1)
                v = 1.0 - (i / (num_phi - 1))
                uvs.append([u, v])

        vertices, uvs = np.array(vertices), np.array(uvs)

        # 应用坐标变换以匹配指定的up_axis
        vertices = apply_transform_to_vertices(vertices, self.transform_matrix)

        faces = self._generate_faces(num_phi, num_theta)

        return {
            "vertices": vertices, 
            "faces": np.array(faces), 
            "uvs": uvs
        }
    
    def _generate_faces(self, num_phi: int, num_theta: int) -> list:
        """生成面片索引"""
        faces = []
        for i in range(num_phi - 1):
            for j in range(num_theta - 1):
                v1, v2 = i * num_theta + j, i * num_theta + (j + 1)
                v3, v4 = (i + 1) * num_theta + (j + 1), (i + 1) * num_theta + j
                faces.extend([[v1, v2, v3], [v1, v3, v4]])
        return faces
    
    def save_mesh_with_uvs(self, mesh_data: Dict[str, np.ndarray], filepath: str):
        """
        保存带UV坐标的网格到OBJ文件
        
        Args:
            mesh_data: 网格数据字典
            filepath: 保存路径
        """
        vertices = mesh_data.get("vertices")
        faces = mesh_data.get("faces")
        uvs = mesh_data.get("uvs")

        if vertices is None or faces is None or uvs is None:
            raise ValueError("网格数据不完整，无法保存")

        with open(filepath, 'w') as f:
            f.write("mtllib texture.mtl\nusemtl Textured\n\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            for vt in uvs:
                f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
            f.write("\n")
            for face in faces:
                f1, f2, f3 = face[0] + 1, face[1] + 1, face[2] + 1
                f.write(f"f {f1}/{f1} {f2}/{f2} {f3}/{f3}\n")

        # 保存材质文件
        mtl_path = os.path.join(os.path.dirname(filepath), "texture.mtl")
        with open(mtl_path, 'w') as f:
            f.write(
                "newmtl Textured\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nmap_Kd texture.png\n"
            )


def create_pytorch3d_mesh(
    mesh_data: Dict[str, np.ndarray], 
    texture_image: Image.Image, 
    device: str
) -> Meshes:
    """
    从mesh数据和纹理图像创建PyTorch3D mesh
    
    Args:
        mesh_data: 网格数据字典
        texture_image: 纹理图像
        device: 计算设备
        
    Returns:
        mesh: PyTorch3D Meshes对象
    """
    vertices = torch.tensor(mesh_data["vertices"], dtype=torch.float32, device=device)
    faces = torch.tensor(mesh_data["faces"], dtype=torch.long, device=device)
    uvs = torch.tensor(mesh_data["uvs"], dtype=torch.float32, device=device)
    
    # 将纹理图像转换为张量
    texture_array = np.array(texture_image.convert("RGB")).astype(np.float32) / 255.0
    texture_tensor = torch.tensor(texture_array, dtype=torch.float32, device=device)
    
    # 创建 TexturesUV 对象
    textures = TexturesUV(
        maps=texture_tensor.unsqueeze(0),
        verts_uvs=uvs.unsqueeze(0),
        faces_uvs=faces.unsqueeze(0)
    )
    
    # 创建 Meshes 对象
    mesh = Meshes(
        verts=[vertices],
        faces=[faces],
        textures=textures
    )
    
    return mesh
