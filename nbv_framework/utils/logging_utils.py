import logging
from pathlib import Path
from loguru import logger
from rich.logging import RichHandler
from rich.traceback import install
from icecream import ic
import lovely_tensors as lt
import pprint
from datetime import datetime

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging(save_dir: str = "logs", rank: int = 0, log_name: str = "train"):
    # 1. 基础准备
    log_path = Path(save_dir)
    if rank == 0:
        log_path.mkdir(parents=True, exist_ok=True)
    
    logger.remove() # 移除默认输出
    
    # 2. 终端输出 (Rich)
    # 仅在 Rank 0 打印到终端，避免多进程刷屏
    if rank == 0:
        logger.add(
            RichHandler(rich_tracebacks=True, markup=True, log_time_format="%H:%M:%S"), 
            format="{message}", 
            level="DEBUG",
            backtrace=True,
            diagnose=True
        )

    # 3. 文件输出 (Loguru)
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = log_path / f"{log_name}_rank{rank}_{time_str}.log"
    
    # 即使是非 0 rank，也记录 WARNING 以上错误到文件，方便排查某张卡死掉的问题
    file_level = "DEBUG" if rank == 0 else "WARNING"
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=file_level,
        rotation="100 MB",
        retention="10 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True, # 训练环境建议开启，可以看到变量值
    )

    # 4. 配置其它增强库
    # Lovely Tensors: 让 Tensor 打印变得好看
    lt.monkey_patch()
    
    # Rich Traceback: 覆盖系统默认异常显示
    install(show_locals=False, suppress=[lt, logging]) # show_locals=True 在 DDP 下可能导致输出过大，建议按需开启

    # Icecream: 配置 Rank 前缀
    def condensed_pprint(obj):
        # 1. 定义一个递归函数来“修剪”列表
        def shrink(item):
            # 如果是列表，且长度大于 1
            if isinstance(item, list) and len(item) > 1:
                # 只取第一个元素，并加上一个说明字符串
                return [shrink(item[0]), f"... <{len(item)-1} more items hidden>"]
            # 如果是字典，递归处理每个 value
            elif isinstance(item, dict):
                return {k: shrink(v) for k, v in item.items()}
            return item

        # 2. 先修剪数据结构（为了不破坏原始 batch，这里先做一次浅拷贝或处理）
        # 注意：由于 batch 可能包含 Tensor，直接 deepcopy 会报错且慢
        # 我们只在打印时修剪
        try:
            # 对于复杂的 batch，我们只针对最外层的 meta 做处理会比较安全
            # 或者使用下面这个通用的修剪逻辑
            shrunk_obj = shrink(obj)
            return pprint.pformat(shrunk_obj, depth=5, width=120, compact=True)
        except Exception:
            # 万一修剪过程报错，退回到普通打印
            return pprint.pformat(obj, depth=5, width=120)
    if rank == 0:
        ic.configureOutput(
            prefix=f'[rank{rank}] ',
            outputFunction=logger.debug,
            argToStringFunction=condensed_pprint
            )
    if rank != 0:
        ic.disable()

    # 5. 拦截标准 logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # 这一步很关键：有些库在 setup_logging 之前就创建了 logger
    # 我们不仅要劫持已有的，还要确保 root logger 的传播
    for name in logging.root.manager.loggerDict:
        child_logger = logging.getLogger(name)
        child_logger.handlers = []
        child_logger.propagate = True

    # 6. 屏蔽第三方库啰嗦的 DEBUG 日志
    # 常见啰嗦的库列表
    noisy_loggers = [
        "urllib3",
        "git",
        "wandb",
        "matplotlib",
        "PIL",
        "stevedore",
        "fsspec",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # 特别针对 wandb，它有时候会有多个子 logger
    for name in logging.root.manager.loggerDict:
        if "wandb" in name or "urllib3" in name:
            logging.getLogger(name).setLevel(logging.WARNING)

    logger.info(f"Logging initialized. Rank: {rank}, File: {log_file}")
    
    return logger
