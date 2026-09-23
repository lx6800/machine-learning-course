import numpy as np  # 导入数值计算库
X = np.array([[1.0], [2.0], [3.0]])  # 准备三条单特征样本
y = np.array([3.0, 5.0, 7.0])  # 准备对应标签
w = np.array([0.0])  # 初始化可学习权重
b = 0.0  # 初始化截距
rate = 0.1  # 设定学习率
for step in range(30):  # 重复执行梯度下降
    pred = X @ w + b  # 计算当前预测
    error = pred - y  # 计算预测减真实的误差
    grad_w = 2 * X.T @ error / len(y)  # 计算平均平方损失对权重的梯度
    grad_b = 2 * error.mean()  # 计算平均平方损失对截距的梯度
    w -= rate * grad_w  # 沿负梯度更新权重
    b -= rate * grad_b  # 沿负梯度更新截距
print(w, b)  # 显示训练结果
