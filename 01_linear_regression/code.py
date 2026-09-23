import numpy as np  # 导入 NumPy
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 导入对照模型
X = np.array([[1.0], [2.0], [3.0], [4.0]])  # 输入特征
y = np.array([3.0, 5.0, 7.0, 9.0])  # 真实标签
A = np.c_[X, np.ones(len(X))]  # 手动增加截距列
w_closed = np.linalg.lstsq(A, y, rcond=None)[0]  # 用稳定最小二乘求闭式解
ridge_penalty = np.diag([0.5, 0.0])  # 只惩罚斜率而不惩罚截距
w_ridge = np.linalg.solve(A.T @ A + ridge_penalty, A.T @ y)  # 求岭回归解
w = np.zeros(X.shape[1])  # 初始化梯度下降权重
b = 0.0  # 初始化梯度下降截距
rate = 0.05  # 设定学习率
for step in range(1000):  # 重复训练
    error = X @ w + b - y  # 计算每条样本的误差
    w -= rate * (2 * X.T @ error / len(y))  # 更新权重
    b -= rate * (2 * error.mean())  # 更新截距
model = LinearRegression().fit(X, y)  # 用 sklearn 训练普通线性回归
ridge = Ridge(alpha=0.5).fit(X, y)  # 用 sklearn 训练岭回归
lasso = Lasso(alpha=0.05).fit(X, y)  # 用 sklearn 训练 Lasso
print('正规方程', w_closed)  # 显示最小二乘解
print('手写梯度下降', w, b)  # 显示迭代解
print('手写岭回归', w_ridge)  # 显示正则化解
print('sklearn', model.coef_, model.intercept_)  # 显示库实现结果
print('Ridge / Lasso', ridge.coef_, lasso.coef_)  # 比较两种正则化
