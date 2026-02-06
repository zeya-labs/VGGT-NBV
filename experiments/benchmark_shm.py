import torch
from torch.utils.data import Dataset, DataLoader
import time
import os
import multiprocessing

# ================= 配置区域 =================
# 模拟 Mesh 在内存中的大小
# 假设 10MB 文件加载后膨胀为 50MB (float32)
# 50MB = 50 * 1024 * 1024 bytes
# float32 = 4 bytes
# 元素数量 = 13,107,200
TENSOR_SHAPE = (131072, 100) 

BATCH_SIZE = 16
NUM_WORKERS = 4
TOTAL_BATCHES = 50  # 测试多少个 Batch (足够多以预热)

# ===========================================

class FakeMeshDataset(Dataset):
    def __init__(self, length=1000):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 模拟读取 Mesh 并转为 Tensor 的过程
        # 这里直接生成随机数据模拟内存占用
        # 注意：生成随机数本身消耗 CPU，但我们要测的是传输(IPC)开销
        data = torch.randn(TENSOR_SHAPE) 
        return data

def run_benchmark(strategy_name):
    print(f"\n[测试开始] 策略: {strategy_name}")
    
    # 1. 设置共享策略
    try:
        torch.multiprocessing.set_sharing_strategy(strategy_name)
    except RuntimeError as e:
        print(f"警告: 无法切换策略 (可能需要重启脚本单独测试): {e}")
        return

    # 2. 构建 DataLoader
    dataset = FakeMeshDataset(length=BATCH_SIZE * TOTAL_BATCHES * 2)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        prefetch_factor=2, # 模拟你的配置
        drop_last=True
    )

    # 3. 预热 (Warmup)
    # 让 Worker 启动并填充队列，消除启动抖动
    print("正在预热 Worker...")
    iterator = iter(loader)
    for _ in range(5):
        _ = next(iterator)

    # 4. 正式计时
    print(f"开始计时 ({TOTAL_BATCHES} Batches)...")
    start_time = time.time()
    
    for i in range(TOTAL_BATCHES):
        _ = next(iterator)
        
    end_time = time.time()
    
    # 5. 计算结果
    total_time = end_time - start_time
    avg_time = total_time / TOTAL_BATCHES
    throughput = (BATCH_SIZE * TOTAL_BATCHES) / total_time
    
    print(f"[结果] {strategy_name}:")
    print(f"  - 总耗时: {total_time:.4f} s")
    print(f"  - 单Batch耗时: {avg_time:.4f} s")
    print(f"  - 吞吐量 (Meshes/sec): {throughput:.2f}")

if __name__ == '__main__':
    # 打印当前环境信息
    print(f"PID: {os.getpid()}")
    print(f"CPU Count: {multiprocessing.cpu_count()}")
    print(f"单 Mesh Tensor 大小: {4 * 131072 * 100 / 1024 / 1024:.2f} MB")
    print(f"单 Batch 数据量: {4 * 131072 * 100 * BATCH_SIZE / 1024 / 1024:.2f} MB")
    
    # --- 测试 1: File System (无 SHM 限制) ---
    # 建议先测这个，因为不用重启
    try:
        run_benchmark('file_system')
    except Exception as e:
        print(f"File System 测试失败: {e}")
        print("提示: 如果报错 'Too many open files'，请在终端运行 'ulimit -n 65535'")

    # --- 测试 2: File Descriptor (默认 SHM) ---
    # 注意：如果你的 docker shm 很小，这里可能会直接报错崩溃
    print("\n------------------------------------------------")
    print("准备测试 file_descriptor (默认 SHM)...")
    print("注意：如果这一步报错 'Bus error'，说明爆显存/共享内存了")
    try:
        run_benchmark('file_descriptor')
    except Exception as e:
        print(f"SHM 测试失败: {e}")