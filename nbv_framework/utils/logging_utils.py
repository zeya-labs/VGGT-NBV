import logging
import sys
from pathlib import Path
from loguru import logger
from rich.logging import RichHandler
from rich.traceback import install
from icecream import ic
import lovely_tensors as lt
from rich.pretty import pretty_repr
from datetime import datetime

class InterceptHandler(logging.Handler):
    """
    将标准 logging 库的日志转发给 loguru
    """
    def emit(self, record):
        # 获取对应的 Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # 查找调用者的栈帧，确保 file/line 准确
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging(save_dir: str = "logs", rank: int = 0, log_name: str = "train"):
    """
    配置完美的 Loguru + Rich + 拦截标准 Logging
    
    Args:
        save_dir: 日志保存目录
        rank: 进程 ID (DDP 模式下使用)
        log_name: 日志文件前缀
    """
    log_path = Path(save_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 0. 重置 Loguru 配置，防止重复添加
    logger.remove()
    
    # 1. 控制台输出 (仅 Rank 0)
    if rank == 0:
        logger.add(
            RichHandler(rich_tracebacks=True, markup=True), 
            format="{message}", 
            level="INFO",
            backtrace=True,
            diagnose=True
        )
    
    # 2. 文件输出 (区分 Rank，避免写入冲突)
    # 策略：Rank 0 记录 DEBUG 全量日志；其他 Rank 仅在 ERROR 时记录（或记录到不同文件）
    # 如果你想所有 Rank 都记录，文件名必须包含 rank
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 2. 生成固定的文件名 (不再使用 Loguru 的 {time} 模板)
    log_file = log_path / f"{log_name}_rank{rank}_{time_str}.log"
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG" if rank == 0 else "WARNING", # 非主卡只记警告以上，节省空间
        rotation="100 MB",     # 文件大小切分
        retention="10 days",   # 保留时间
        compression="zip",     # 压缩历史日志
        enqueue=True,          # 关键：异步写入，不阻塞训练主循环
        backtrace=True,
        diagnose=True,
    )

    # 3. 压制第三方库的噪声 (即使是 DEBUG 级别也不想看到的)
    # 这些库在 DEBUG 模式下非常吵，建议设置为 INFO 或 WARNING
    noisy_modules = ["PIL", "matplotlib", "numba", "fsspec", "asyncio"]
    for module_name in noisy_modules:
        logging.getLogger(module_name).setLevel(logging.WARNING)

    # 4. 配置工具链
    lt.monkey_patch() # 让 Tensor 打印变好看
    install(show_locals=True, suppress=[lt, logging]) # Rich 全局异常捕获
    
    # 配置 IceCream 使用 Loguru 输出
    # 如果非 Rank 0，禁用 ic 输出以减少日志噪音
    if rank == 0:
        ic.configureOutput(
            argToStringFunction=pretty_repr, 
            outputFunction=lambda x: logger.debug(x),
            prefix="🍦 "
        )
    else:
        ic.disable()

    # --- 5. 核心魔法：全自动劫持标准 Logging ---
    
    # A. 劫持根 Logger
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # B. 遍历并劫持现有 Logger
    # 使用 loggerDict.items() 并判断类型，避免实例化 PlaceHolder
    for name, logger_obj in logging.root.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger):
            logger_obj.handlers = [] # 清空原有 Handler (如 StreamHandler)
            logger_obj.propagate = True # 开启传播，让 InterceptHandler 接管
            # 也可以在这里强制设置 level，但通常保留原库的 level 逻辑更好

    logger.info(f"Logging setup complete. Rank: {rank}, Log file: {log_file}")
    
    return logger