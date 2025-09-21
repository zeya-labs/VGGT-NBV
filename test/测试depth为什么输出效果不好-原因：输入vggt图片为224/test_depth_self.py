import os
import sys
import glob
import torch
from PIL import Image
from torchvision import transforms

sys.path.append("vggt/")

from visual_util import predictions_to_glb 
from vggt.models.vggt import VGGT   
from vggt.utils.load_fn import load_and_preprocess_images  # 图像加载和预处理函数
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # 姿态编码转换函数
from vggt.utils.geometry import unproject_depth_map_to_point_map  # 深度图转点云函数

from nbv_framework import VGGTWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing and loading VGGT model...")
model = VGGTWrapper(
        model_name="facebook/VGGT-1B",
        device=device
    )
# model = VGGT.from_pretrained("facebook/VGGT-1B") 

# 将模型设置为评估模式（关闭dropout等训练时的随机性）
# model.eval()
# # 将模型移动到指定的计算设备上
# model = model.to(device)
print("Model loaded successfully.")


target_dir = "/mnt/sdb/chenmohan/VGGT-NBV/vggt/input_images_20250915_200421_843592"
image_names = glob.glob(os.path.join(target_dir, "images", "*"))
image_names = sorted(image_names)  # 按文件名排序确保处理顺序一致
print(image_names)
print(f"Found {len(image_names)} images")
if len(image_names) == 0:
    raise ValueError("No images found. Check your upload.")

# 加载并预处理图像，然后移动到GPU
# images = load_and_preprocess_images(image_names).to(device)
# 直接读取
images = torch.stack([transforms.ToTensor()(Image.open(image_name)) for image_name in image_names]).to(device)
print(f"Preprocessed images shape: {images.shape}")

# 运行模型推理
print("Running inference...")
# 根据GPU能力选择合适的数据类型（较新的GPU支持bfloat16）
# dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# 在无梯度模式下运行推理以节省内存
with torch.no_grad():
    # 使用自动混合精度加速推理
    # with torch.cuda.amp.autocast(dtype=dtype):
    predictions = model.reconstruct_and_evaluate(images)

# # 将姿态编码转换为外参和内参矩阵
# print("Converting pose encoding to extrinsic and intrinsic matrices...")
# print(f"Pose enc shape: {predictions['pose_enc'].shape}")
# # 从姿态编码中提取相机的外参（位置和方向）和内参（焦距等）
# extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
# predictions["extrinsic"] = extrinsic
# predictions["intrinsic"] = intrinsic

# # 将PyTorch张量转换为NumPy数组以便后续处理
for key in predictions.keys():
    if isinstance(predictions[key], torch.Tensor):
        # 移动到CPU并移除批次维度
        predictions[key] = predictions[key].cpu().numpy().squeeze(0)
# predictions['pose_enc_list'] = None  # 移除不需要的pose_enc_list

# # 从深度图生成世界坐标点云
# print("Computing world points from depth map...")
# depth_map = predictions["depth"]  # 深度图形状: (S, H, W, 1)
# print(f"Depth map shape: {depth_map.shape}")
# print(f"Extrinsic shape: {predictions['extrinsic'].shape}")
# print(f"Intrinsic shape: {predictions['intrinsic'].shape}")
# # 使用深度图和相机参数将2D像素反投影到3D世界坐标
# world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
# predictions["world_points_from_depth"] = world_points

# 清理GPU内存
torch.cuda.empty_cache()

# 构建GLB文件名（包含所有参数信息以避免重复计算）
glbfile = os.path.join(
    target_dir,
    f"glbscene_gradio.glb",
)

# 将预测结果转换为GLB 3D模型格式
glbscene = predictions_to_glb(
    predictions,
    conf_thres=80,          # 置信度阈值
    filter_by_frames="All",  # 帧过滤
    mask_black_bg=True,    # 黑色背景遮罩
    mask_white_bg=False,    # 白色背景遮罩
    show_cam=False,              # 显示相机
    mask_sky=False,              # 天空遮罩
    target_dir=target_dir,          # 目标目录
    # prediction_mode="Depthmap and Camera Branch", # 预测模式
    prediction_mode="Pointmap Branch", # 预测模式
)
# 导出GLB文件
glbscene.export(file_obj=glbfile)