import torch
import h5py
import numpy as np
import sys
import os

# 把模型定义所在目录加入路径
sys.path.append(os.path.join(os.getcwd(), 'torch', 'rnnoise'))

# 导入 RNNoise 类
from rnnoise import RNNoise

# 1. 加载 384 维教师模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ★★★ 唯一的核心修改：显式传入 gru_size=384 ★★★
model = RNNoise(gru_size=384) 

weight_path = 'models/rnnoise10Gb_15.pth'

if not os.path.exists(weight_path):
    print(f"❌ 找不到权重文件 {weight_path}")
    sys.exit(1)

print(f"加载权重: {weight_path}")
checkpoint = torch.load(weight_path, map_location=device)

# 兼容不同的保存格式
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
model.to(device)

# 2. 读取 100 万帧数据
data_path = '/home/rnnoise/22bands_rnnoise_old/rnnoise/training/training.h5'
print(f"加载数据: {data_path}")
with h5py.File(data_path, 'r') as f:
    x_all = f['data'][:, :42]   # 只取前 42 维特征

# 3. 分批推理
batch_size = 4096
teacher_gains_list = []
teacher_vads_list = []

print("开始推理，提取软标签...")
with torch.no_grad():
    for i in range(0, len(x_all), batch_size):
        batch_x = torch.tensor(x_all[i:i+batch_size], dtype=torch.float32).to(device)
        
        # 为了防止 PyTorch 的 Conv1d 维度报错，这里加一层安全处理
        # 如果原版 forward 需要 (batch, seq_len, 42)，直接用；如果需要三维，加 unsqueeze
        if len(batch_x.shape) == 2:
            batch_x = batch_x.unsqueeze(1)  # 变成 (batch, 1, 42)
            
        outputs = model(batch_x)
        
        # RNNoise 通常返回 (gains, vad)，打印一次确认形状
        if i == 0:
            print(f"模型输出类型: {type(outputs)}")
            if isinstance(outputs, (tuple, list)):
                print(f"输出数量: {len(outputs)}")
                for idx, out in enumerate(outputs):
                    print(f"  输出[{idx}] shape: {out.shape}")
            else:
                print(f"输出 shape: {outputs.shape}")
        
        # 根据实际输出调整
        if isinstance(outputs, (tuple, list)):
            gains = outputs[0]
            vad = outputs[1] if len(outputs) > 1 else None
        else:
            gains = outputs
            vad = None
        
        # 把可能带有的冗余维度 squeeze 掉
        gains = gains.squeeze(1) if len(gains.shape) > 2 else gains
        teacher_gains_list.append(gains.cpu().numpy())
        
        if vad is not None:
            vad = vad.squeeze(1) if len(vad.shape) > 2 else vad
            teacher_vads_list.append(vad.cpu().numpy())
        
        if (i // batch_size) % 50 == 0:
            print(f"进度: {i}/{len(x_all)}")

# 4. 拼接并保存
teacher_gains = np.concatenate(teacher_gains_list, axis=0)
print(f"Teacher Gains shape: {teacher_gains.shape}")

save_dir = '/home/rnnoise/22bands_rnnoise_old/rnnoise/training/'
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'teacher_gains.npy'), teacher_gains)
print(f"✅ teacher_gains.npy 已保存到 {save_dir}")

if teacher_vads_list:
    teacher_vads = np.concatenate(teacher_vads_list, axis=0)
    print(f"Teacher VADs shape: {teacher_vads.shape}")
    np.save(os.path.join(save_dir, 'teacher_vads.npy'), teacher_vads)
    print(f"✅ teacher_vads.npy 已保存到 {save_dir}")
else:
    print("⚠️ 模型未返回 VAD 输出，仅保存了 gains。请检查模型定义。")