import numpy as np  # 导入 NumPy
from sklearn.svm import SVC  # 导入核 SVM
X = np.array([[-2.0], [-1.0], [1.0], [2.0]])  # 准备一维样本
y = np.array([1, -1, -1, 1])  # 设定两端同类、中间同类的非线性标签
def poly_kernel(A, B):  # 手写二次多项式核
    return (1 + A @ B.T) ** 2  # 直接计算隐式特征空间内积
def rbf_kernel(A, B, length=1.0):  # 手写 RBF 核
    sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)  # 计算两两平方距离
    return np.exp(-sq / (2 * length ** 2))  # 把距离转成相似度
K = poly_kernel(X, X)  # 计算训练 Gram 矩阵
eigenvalues = np.linalg.eigvalsh(K)  # 检查矩阵特征值
model = SVC(kernel='poly', degree=2, gamma=1, coef0=1, C=10).fit(X, y)  # 训练二次核 SVM
print('Gram 矩阵', K)  # 显示手写核矩阵
print('最小特征值', eigenvalues.min())  # 显示半正定检查结果
print('RBF 相似度', rbf_kernel(X, X))  # 显示距离型核
print('sklearn 分类', model.predict(X))  # 显示训练点分类结果
