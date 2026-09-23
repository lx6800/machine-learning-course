import numpy as np  # 导入 NumPy
from sklearn.linear_model import LogisticRegression  # 导入分类器
from sklearn.model_selection import train_test_split  # 导入数据划分工具
from sklearn.metrics import zero_one_loss  # 导入零一损失
rng = np.random.default_rng(7)  # 固定随机种子以复现实验
X = rng.normal(size=(200, 3))  # 生成三维特征样本
y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(int)  # 根据线性规则生成标签
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=7)  # 划分训练与测试集
model = LogisticRegression().fit(Xtrain, ytrain)  # 仅在训练集上拟合模型
ein = zero_one_loss(ytrain, model.predict(Xtrain))  # 计算训练错误率
etest = zero_one_loss(ytest, model.predict(Xtest))  # 计算独立测试错误率
M = 10  # 假设训练前只考虑十个固定候选模型
delta = 0.05  # 设置失败概率
bound = np.sqrt(np.log(M / delta) / (2 * len(Xtrain)))  # 计算有限假设类的单侧复杂度项
print('训练误差', ein, '测试误差', etest)  # 显示经验结果
print('示例复杂度项', bound)  # 显示理论项并注意 M 的前提
