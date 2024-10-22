import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 指定.npy文件路径
file_path1 = r"C:\Users\14168\1\Python\pythonProject\Informer2020\results\informer_custom_ftM_sl10_ll5_pl20_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1\true.npy"
file_path2 = r"C:\Users\14168\1\Python\pythonProject\Informer2020\results\informer_custom_ftM_sl10_ll5_pl20_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0\pred.npy"

# 使用NumPy加载.npy文件
data1 = np.load(file_path1)
data2 = np.load(file_path2)

# 打印数据的形状
print("Shape of data1 (true values):", data1.shape)
print("Shape of data2 (predicted values):", data2.shape)

# 选择要访问的特征索引，确保索引在 [0, num_features - 1] 范围内
feature_idx = 2  # 假设你要访问第4个特征，索引为3

# 确保索引在合法范围内
if feature_idx < data1.shape[2]:
    true_value = []
    pred_value = []

    # 迭代前24个序列的第 feature_idx 个特征
    for i in range(24):
        for t in range(10):  # 迭代每个时间步
            true_value.append(data1[i][t][feature_idx])
            pred_value.append(data2[i][t][feature_idx])

    # 打印内容
    print("True values:", true_value)
    print("Predicted values:", pred_value)

    # 创建DataFrame并保存为CSV
    if true_value and pred_value:
        df = pd.DataFrame({'real': true_value, 'pred': pred_value})
        df.to_csv('results.csv', index=False)
        print("Results saved to results.csv.")
    else:
        print("No values to save.")
else:
    print(f"Feature index {feature_idx} is out of bounds for data with shape {data1.shape}")
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(df['real'], df['pred'])
mae = mean_absolute_error(df['real'], df['pred'])
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
# 可视化结果
# 读取保存的 CSV 文件
df = pd.read_csv('results.csv')

# 绘制图表，展示真实值和预测值
plt.figure(figsize=(12, 6))
plt.plot(df['real'], label='True Values', color='b', linestyle='-', marker='o', markersize=4)
plt.plot(df['pred'], label='Predicted Values', color='r', linestyle='--', marker='x', markersize=4)
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('True vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()