import numpy as np  # 导入 NumPy
from sklearn.svm import SVC  # 导入支持向量机
X = np.array([[-2.0], [-1.0], [1.0], [2.0]])  # 准备一维可分样本
y = np.array([-1.0, -1.0, 1.0, 1.0])  # 准备正负标签
w = np.array([0.0])  # 初始化手写线性分类器权重
b = 0.0  # 初始化截距
rate = 0.02  # 设定学习率
C = 2.0  # 设定间隔违例惩罚
for step in range(3000):  # 使用次梯度法训练软间隔目标
    margin = y * (X @ w + b)  # 计算每条样本的功能间隔
    active = margin < 1  # 找到产生铰链损失的样本
    grad_w = w - C * np.sum(y[active, None] * X[active], axis=0)  # 计算权重次梯度
    grad_b = -C * np.sum(y[active])  # 计算截距次梯度
    w -= rate * grad_w  # 更新权重
    b -= rate * grad_b  # 更新截距
model = SVC(kernel='linear', C=C).fit(X, y)  # 用 sklearn 训练线性 SVM
print('手写决策', np.sign(X @ w + b))  # 显示手写分类结果
print('sklearn 支持向量', model.support_vectors_.ravel())  # 显示支持向量
print('sklearn 决策', model.predict(X))  # 显示库分类结果
