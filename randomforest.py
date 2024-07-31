import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv('data.csv')

# 显示前几行数据
print(df.head())

# 查看数据描述信息
print(df.describe())

# 检查缺失值
print(df.isnull().sum())

# 如果有缺失值，可以选择填补缺失值，例如使用均值填补
df.fillna(df.mean(), inplace=True)

# 分离特征和目标变量
# 根据相关性矩阵选择相关性较高的特征
selected_features = ['Milk Yield', 'Protein Percentage']
X = df[selected_features]
y = df['Lactation days']

# 查看特征和目标变量之间的相关性
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# 归一化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 设置参数网格
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

# 使用网格搜索和交叉验证来找到最佳参数
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数
print(f'Best parameters: {grid_search.best_params_}')

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred_best = best_model.predict(X_test_scaled)

# 评估改进后模型的性能
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
print(f'Improved Mean Absolute Error: {mae_best}')
print(f'Improved Mean Squared Error: {mse_best}')

# 使用改进后的模型进行预测
# 假设 new_data 是新的数据
new_data = np.array([[41,3.51],[39,3.51],[44,3.31]])  # 替换为实际值
new_data_scaled = scaler.transform(new_data)
predictions_best = best_model.predict(new_data_scaled)
print(f'Improved Predicted 泌乳天数: {predictions_best}')
