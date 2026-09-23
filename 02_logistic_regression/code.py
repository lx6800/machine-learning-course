import numpy as np  # 导入 NumPy
from sklearn.linear_model import LogisticRegression  # 导入库实现
X = np.array([[-2.0], [-1.0], [1.0], [2.0]])  # 准备一维分类特征
y = np.array([0.0, 0.0, 1.0, 1.0])  # 准备二元标签
def sigmoid(z):  # 定义 sigmoid 函数
    z = np.clip(z, -40, 40)  # 限制指数输入以避免溢出
    return 1 / (1 + np.exp(-z))  # 返回正类概率
w = np.zeros(X.shape[1])  # 初始化权重
b = 0.0  # 初始化截距
rate = 0.1  # 设定学习率
for step in range(1000):  # 反复进行梯度下降
    p = sigmoid(X @ w + b)  # 计算每条样本的正类概率
    w -= rate * (X.T @ (p - y) / len(y))  # 更新权重
    b -= rate * np.mean(p - y)  # 更新截距
p = sigmoid(X @ w + b)  # 计算训练后的概率
loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))  # 计算交叉熵
model = LogisticRegression(C=1000).fit(X, y)  # 训练弱正则化的 sklearn 模型
print('手写概率', p, '交叉熵', loss)  # 显示手写结果
print('sklearn 概率', model.predict_proba(X)[:, 1])  # 显示库结果
