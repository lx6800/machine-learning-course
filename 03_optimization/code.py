import numpy as np  # 导入 NumPy
from scipy.optimize import minimize  # 导入通用约束优化器
x = 0.0  # 初始化一个标量参数
rate = 0.1  # 设定梯度步长
for step in range(100):  # 迭代投影梯度法
    gradient = 2 * (x - 3)  # 计算目标 (x-3)^2 的导数
    x = x - rate * gradient  # 先走无约束下降步
    x = min(x, 1.0)  # 再投影回可行域 x<=1
result = minimize(lambda z: (z[0] - 3) ** 2, [0.0], constraints={'type': 'ineq', 'fun': lambda z: 1 - z[0]})  # 用 scipy 求同一约束问题
print('手写投影法', x)  # 输出手写解
print('scipy 对照', result.x[0])  # 输出库实现解
