from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

# print(model.predict(data_X[:4,:]))
# print(data_y[:4])

# print(model.coef_)          # 模型系数
# print(model.intercept_)     # 模型截距
# print(model.get_params())
print(model.score(data_X,data_y))  # R^2 coefficient of determination


X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
plt.scatter(X, y)
plt.show()
