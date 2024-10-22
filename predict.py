import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图
from exp.exp_informer import Exp_Informer
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler

# 设置参数
model_path = r"C:\Users\14168\1\Python\pythonProject\Informer2020\checkpoints\informer_custom_ftM_sl10_ll5_pl20_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0\model.pt"  # 替换为你保存的模型路径
data_path = r"C:\Users\14168\1\Python\pythonProject\LSTM\all_shots_data.csv"  # 新数据的路径
features_to_predict = ['x', 'y', 'z']  # 只预测 x, y, z
cols_to_use = ['x', 'y', 'z', 'vx', 'vy', 'vz']  # 使用的输入特征

# 1. 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)

# 2. 加载并预处理新数据
df = pd.read_csv(data_path)

# 将数据按 shot_id 分组
grouped = df.groupby('shot_id')

for shot_id, group in grouped:
    print(f"Processing shot_id: {shot_id}")

    # 选择需要的特征列
    group = group.reset_index(drop=True)
    data = group[cols_to_use]

    # 标准化输入数据（使用与训练时相同的标准化）
    scaler = StandardScaler()
    scaler.fit(data)  # 假设你保存了训练时使用的标准化参数，你应该加载相同的参数
    data_normalized = scaler.transform(data)

    # 3. 准备输入数据为张量格式，选取前十个样本作为基准
    seq_len = 20  # 使用前10个样本进行预测
    if len(data_normalized) < seq_len:
        print(f"Shot {shot_id} has insufficient data for prediction. Skipping.")
        continue

    input_data = data_normalized[:seq_len]  # 选取前10个样本
    input_tensor = torch.tensor(input_data, dtype=torch.float).unsqueeze(0).to(device)  # 形状为 [1, seq_len, num_features]

    # 4. 生成时间特征
    group_stamp = group[['timestamp']][:len(data)]  # 选择 timestamp 列
    group_stamp['timestamp'] = pd.to_datetime(group_stamp['timestamp'])  # 转换为 datetime 类型

    # 使用 time_features 生成时间特征
    data_stamp = time_features(group_stamp, timeenc=1, freq='s')  # 使用 timeenc=1 生成秒级时间特征
    data_stamp_tensor = torch.tensor(data_stamp[:seq_len], dtype=torch.float).unsqueeze(0).to(device)  # Encoder 时间特征

    # 5. 准备 Decoder 的输入数据
    pred_len = 10  # 预测未来 10 个时间步
    decoder_start = data_normalized[seq_len - 1:seq_len]  # 使用最后一个时间步作为 Decoder 的起始输入
    decoder_input = np.repeat(decoder_start, pred_len, axis=0)  # 将这个时间步重复 pred_len 次
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float).unsqueeze(0).to(
        device)  # 形状为 [1, pred_len, num_features]

    # 6. 生成 Decoder 的时间特征
    future_stamp = pd.date_range(group_stamp['timestamp'].iloc[seq_len - 1], periods=pred_len + 1, freq='S')  # 创建未来的时间戳
    future_stamp = future_stamp[1:]  # 删除起点
    future_stamp_df = pd.DataFrame({'timestamp': future_stamp})  # 保持列名为 timestamp
    future_stamp_data = time_features(future_stamp_df, timeenc=1, freq='s')  # 修改为支持的频率 's'
    future_stamp_tensor = torch.tensor(future_stamp_data, dtype=torch.float).unsqueeze(0).to(device)  # Decoder 时间特征

    # 7. 进行预测
    model.eval()
    with torch.no_grad():
        # 输出的形状：[1, pred_len, num_features]
        predictions = model(input_tensor, data_stamp_tensor, decoder_input_tensor, future_stamp_tensor)

    # 8. 处理输出，提取 x, y, z 的预测值
    predictions_numpy = predictions.cpu().numpy()  # 转为 NumPy 格式
    predictions_xyz = predictions_numpy[:, :, :3]  # 只取前三个特征（x, y, z）

    # 逆标准化预测值，只对 x, y, z
    predictions_xyz_original = scaler.inverse_transform(
        np.hstack((predictions_xyz.reshape(-1, 3), np.zeros((predictions_xyz.shape[1], 3))))
    )[:, :3]

    # 9. 准备真实值进行对比
    # 取前 10 个样本作为基准真实值
    baseline_values_xyz = data[:seq_len][features_to_predict].reset_index(drop=True)

    # 取未来的真实值进行对比
    real_values_xyz = data[seq_len:seq_len + pred_len][features_to_predict].reset_index(drop=True)

    # 10. 3D 可视化对比
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制基准真实轨迹 (前 10 个)
    ax.plot(baseline_values_xyz['x'], baseline_values_xyz['y'], baseline_values_xyz['z'], label='Baseline Trajectory',
            color='blue', marker='o')

    # 绘制真实未来轨迹 (x, y, z)
    ax.plot(real_values_xyz['x'], real_values_xyz['y'], real_values_xyz['z'], label='True Future Trajectory',
            color='green', marker='^')

    # 绘制预测轨迹 (x, y, z)
    ax.plot(predictions_xyz_original[:, 0], predictions_xyz_original[:, 1], predictions_xyz_original[:, 2],
            label='Predicted Future Trajectory', color='red', linestyle='--', marker='x')

    # 设置轴标签
    ax.set_xlabel('X (horizontal distance)')
    ax.set_ylabel('Y (height)')
    ax.set_zlabel('Z (depth/left-right)')
    plt.title(f'Shot ID {shot_id}: 3D Baseline, True Future, and Predicted Future Trajectory')
    plt.legend()
    plt.show()