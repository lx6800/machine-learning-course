import numpy as np  # 导入 NumPy
from sklearn.gaussian_process import GaussianProcessRegressor  # 导入 GP 回归器
from sklearn.gaussian_process.kernels import RBF, ConstantKernel  # 导入核组件
X = np.array([[0.0], [1.0], [2.0]])  # 准备训练位置
y = np.array([0.0, 1.0, 0.0])  # 准备观测值
test = np.array([[0.5], [1.5]])  # 准备待预测位置
noise = 0.05  # 设置观测噪声方差
def kernel(A, B, length=1.0):  # 手写 RBF 协方差核
    sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)  # 计算点对平方距离
    return np.exp(-sq / (2 * length ** 2))  # 返回协方差矩阵
K = kernel(X, X) + noise * np.eye(len(X))  # 构造含观测噪声的训练协方差
Ks = kernel(test, X)  # 计算测试与训练的交叉协方差
alpha = np.linalg.solve(K, y)  # 解线性系统而不显式求逆
mean = Ks @ alpha  # 计算后验预测均值
v = np.linalg.solve(K, Ks.T)  # 为方差公式解线性系统
variance = np.diag(kernel(test, test)) - np.sum(Ks * v.T, axis=1)  # 计算潜在函数后验方差
model = GaussianProcessRegressor(kernel=ConstantKernel(1.0, constant_value_bounds='fixed') * RBF(1.0, length_scale_bounds='fixed'), alpha=noise, optimizer=None, normalize_y=False).fit(X, y)  # 训练同参数的 sklearn GP
sk_mean, sk_std = model.predict(test, return_std=True)  # 获取库实现的均值和标准差
print('手写均值与方差', mean, variance)  # 显示手写计算
print('sklearn 均值与方差', sk_mean, sk_std ** 2)  # 显示库实现对照
