import numpy as np
import lovely_numpy as ln
from icecream import ic

arr = np.random.randn(2, 3)
ic(ln.lo(arr)) # 使用 ln.lo() 包装