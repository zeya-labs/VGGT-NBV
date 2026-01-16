from rich import print
from icecream import ic
import lovely_tensors
import torch
from rich.pretty import pretty_repr

ic.configureOutput(argToStringFunction=pretty_repr)

import time
def my_prefix():
    return f"DEBUG {time.strftime('%H:%M:%S')} |> "

ic.configureOutput(prefix=my_prefix)

# ic.configureOutput(includeContext=False)

lovely_tensors.monkey_patch()

# 造一个稍微复杂点的数据
data = {
    "用户ID": 10086,
    "姓名": "测试员",
    "权限列表": ["读取", "写入", "删除", "管理"],
    "配置信息": {
        "主题": "暗黑模式",
        "通知": True,
        "历史记录": [
            {"时间": "2023-01-01", "IP": "192.168.1.1"},
            {"时间": "2023-01-02", "IP": "192.168.1.2"},
        ]
    },
    "tensor数据": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

}

a=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(data)
ic(data)
ic(data["tensor数据"])
ic(a)

from rich.traceback import install
install(show_locals=True)  # 只要加这一行！

# 你的烂代码
def divide(x, y):
    return x / y

divide(10, 0) # 触发 ZeroDivisionError


from rich import inspect
import requests

r = requests.get('https://www.baidu.com')

# 看看这个对象里到底有啥
inspect(r, methods=True)    

class helloworld:
    def __init__(self):
        self.a = 123
        self.b = "hello"
        self.c = [1, 2, 3]

hello = helloworld()

inspect(hello, methods=True)

from rich.console import Console
from rich.table import Table

table = Table(title="服务器状态")

table.add_column("IP 地址", style="cyan")
table.add_column("状态", style="green")
table.add_column("延迟", justify="right")

table.add_row("192.168.1.1", "在线", "12ms")
table.add_row("192.168.1.2", "[red]离线[/red]", "-") # 支持直接写颜色标记

console = Console()
console.print(table)