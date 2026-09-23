import numpy as np  # 导入 NumPy
from sklearn.kernel_ridge import KernelRidge  # 导入核岭回归
X = np.array([[0.0], [1.0], [2.0]])  # 准备输入点
y = np.array([0.0, 1.0, 0.0])  # 准备非线性目标
length = 0.8  # 设置 RBF 核长度尺度
lam = 0.1  # 设置岭正则强度
def kernel(A, B):  # 定义手写 RBF 核
    sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)  # 计算两组点的平方距离
    return np.exp(-sq / (2 * length ** 2))  # 返回相似度矩阵
K = kernel(X, X)  # 计算训练 Gram 矩阵
alpha = np.linalg.solve(K + lam * np.eye(len(X)), y)  # 求解核岭系数
test = np.array([[0.5], [1.5]])  # 准备新输入
pred = kernel(test, X) @ alpha  # 用训练点核展开预测
model = KernelRidge(alpha=lam, kernel='rbf', gamma=1 / (2 * length ** 2)).fit(X, y)  # 训练 sklearn 对照模型
print('手写系数', alpha)  # 显示系数
print('手写预测', pred)  # 显示手写结果
print('sklearn 预测', model.predict(test))  # 显示库实现结果
