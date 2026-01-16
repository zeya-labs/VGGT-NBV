import sys
import logging
from pathlib import Path
from loguru import logger
from rich.logging import RichHandler
from rich.traceback import install
from icecream import ic
import lovely_tensors as lt
from rich.pretty import pretty_repr

# --- 辅助类：拦截标准 logging 转发给 loguru ---
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # 获取对应的 Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用者的栈帧
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging(save_dir: str = "logs", rank: int = 0):
    # 0. 如果是多卡训练的非主进程，可以直接禁用 logger (可选)
    # if rank != 0:
    #     logger.remove()
    #     return logger

    # --- 1. 基础配置 ---
    log_path = Path(save_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 清除所有已存在的 handler
    logger.remove()
    
    # --- 2. 控制台输出 (Rich) ---
    # 只有主进程才在控制台打印，避免多卡刷屏
    if rank == 0:
        logger.add(
            RichHandler(
                rich_tracebacks=True, 
                show_path=True, 
                markup=True,
                enable_link_path=True # 允许终端点击路径跳转
            ),
            format="{message}",
            level="DEBUG",
        )
    
    # --- 3. 文件输出 ---
    # 增加 retention(保留10天) 和 compression(zip压缩)
    logger.add(
        log_path / "train_{time:YYYY-MM-DD_HH-mm-ss}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="500 MB",
        retention="10 days", # 自动清理旧日志
        compression="zip",   # 自动压缩旧日志
        enqueue=True,        # 异步写入，线程安全，防止阻塞训练循环
    )

    # --- 4. 拦截标准库 logging (关键) ---
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # --- 5. 增强工具配置 ---
    lt.monkey_patch() 
    
    # Icecream 配置：
    # 1. 使用 Rich 的 pretty print
    # 2. 将 ic() 的输出重定向到 logger.debug，这样日志文件里也能看到 ic 的打印！
    ic.configureOutput(
        argToStringFunction=pretty_repr,
        outputFunction=lambda x: logger.debug(x), 
        includeContext=True
    )
    
    # 全局异常捕获 (Rich Traceback)
    # suppress=[lt] 是为了报错时隐藏 lovely_tensors 内部的栈帧，让报错更清爽
    install(show_locals=True, suppress=[lt]) 
    
    return logger

# 使用示例
if __name__ == "__main__":
    log = setup_logging()
    
    import torch
    
    log.info("Starting training...")
    
    # 测试 1: icecream 是否进入文件
    x = {"a": 1, "b": [1, 2, 3]}
    ic(x)  # 这行现在会以 DEBUG 级别同时出现在控制台和日志文件里
    
    # 测试 2: lovely-tensors
    tensor = torch.randn(2, 3)
    log.info(f"Tensor info: {tensor}") # 会显示 lt 的简略格式
    
    # 测试 3: 第三方库 logging 拦截
    logging.warning("This is a warning from standard logging") # 应该变成 Loguru 格式
    
    # 测试 4: 报错
    try:
        _ = 1 / 0
    except Exception:
        log.exception("Something went wrong")