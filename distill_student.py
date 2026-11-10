import tensorflow as tf
import h5py
import numpy as np
import os

from tensorflow.keras.layers import Input, Dense, GRU, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import Constraint

# ================= 定义模型结构 =================
class WeightClip(Constraint):
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return tf.clip_by_value(p, -self.c, self.c)

def build_student_model():
    input_features = Input(shape=(42,))
    tmp = Dense(24, activation='tanh')(input_features)
    
    # ★ 核心修复 1：把 (batch, 24) 变成 (batch, 1, 24) 以适应 GRU 的 3D 输入
    tmp = Reshape((1, 24))(tmp)
    
    reg = 1e-5
    constraint = WeightClip(0.5)
    
    # ★ 必须带上 unroll=True, reset_after=False，确保 C 端推理兼容
    tmp = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru',
              unroll=True, stateful=False, reset_after=False, kernel_regularizer=regularizers.l2(reg), kernel_constraint=constraint)(tmp)
    tmp = GRU(48, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru',
              unroll=True, stateful=False, reset_after=False, kernel_regularizer=regularizers.l2(reg), kernel_constraint=constraint)(tmp)
    tmp = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru',
              unroll=True, stateful=False, reset_after=False, kernel_regularizer=regularizers.l2(reg), kernel_constraint=constraint)(tmp)
    
    # ★ 核心修复 2：把 3D 的 (batch, 1, 96) 拍平回 2D 的 (batch, 96)
    tmp = Flatten()(tmp)
    
    gains = Dense(22, activation='sigmoid', name='denoise_output')(tmp)
    vad = Dense(1, activation='sigmoid', name='vad_output')(tmp)
    
    return Model(inputs=input_features, outputs=[gains, vad])

# ================= 配置区 =================
DATA_PATH = './training/training.h5'
STUDENT_WEIGHTS = './training/weights.hdf5'
TEACHER_GAINS = './training/teacher_gains.npy'
TEACHER_VADS = './training/teacher_vads.npy'

T = 4.0          # 温度参数
ALPHA = 0.4      # 硬标签权重
BETA = 0.6       # 软标签权重

# ================= 加载模型 =================
print("构建并加载 Student 老模型 (24/48/96)...")
student = build_student_model()
student.load_weights(STUDENT_WEIGHTS)
print("✅ 学生模型权重加载成功！")

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# ================= 加载数据 =================
print("加载数据集和软标签...")
with h5py.File(DATA_PATH, 'r') as f:
    data = f['data'][:]
    x = data[:, :42]             # 42维特征
    y_gains = data[:, 42:64]     # 22维增益
    y_vad = data[:, 86:87]       # 1维VAD

t_gains = np.load(TEACHER_GAINS)
t_vads = np.load(TEACHER_VADS)

dataset = tf.data.Dataset.from_tensor_slices((x, y_gains, y_vad, t_gains, t_vads))\
    .shuffle(10000).batch(1024).prefetch(tf.data.AUTOTUNE)
print("✅ 数据加载完毕！")

# ================= 蒸馏训练步 =================
@tf.function
def train_step(inputs, true_gains, true_vad, soft_teacher_gains, soft_teacher_vad):
    with tf.GradientTape() as tape:
        # 学生输出
        student_gains, student_vad = student(inputs, training=True)

        # 1. 硬标签损失
        loss_hard_gains = tf.reduce_mean(tf.square(student_gains - true_gains))
        loss_hard_vad = tf.reduce_mean(tf.keras.losses.binary_crossentropy(true_vad, student_vad))
        
        # 2. 软标签损失
        loss_soft_gains = tf.reduce_mean(tf.square(student_gains - soft_teacher_gains))
        
        # VAD 带温度 T 的交叉熵
        soft_teacher_vad_T = tf.sigmoid(tf.math.log(soft_teacher_vad / (1.0 - soft_teacher_vad + 1e-7)) / T)
        soft_student_vad_T = tf.sigmoid(tf.math.log(student_vad / (1.0 - student_vad + 1e-7)) / T)
        loss_soft_vad = tf.reduce_mean(tf.keras.losses.binary_crossentropy(soft_teacher_vad_T, soft_student_vad_T)) * (T**2)

        # 3. 总 Loss
        total_loss = ALPHA * (loss_hard_gains + loss_hard_vad) + BETA * (loss_soft_gains + loss_soft_vad)
        
    grads = tape.gradient(total_loss, student.trainable_variables)
    optimizer.apply_gradients(zip(grads, student.trainable_variables))
    return total_loss

# ================= 开始训练 =================
print("🚀 开始蒸馏学生模型...")
EPOCHS = 15
for epoch in range(EPOCHS):
    total_loss_val = 0
    steps = 0
    for batch_x, batch_gains, batch_vad, t_g, t_v in dataset:
        loss = train_step(batch_x, batch_gains, batch_vad, t_g, t_v)
        total_loss_val += loss.numpy()
        steps += 1
        if steps % 200 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Step {steps}, Loss: {loss.numpy():.4f}")
    print(f"✅ Epoch {epoch+1} 结束, 平均 Loss: {total_loss_val/steps:.4f}")

# 保存最终蒸馏后的权重
student.save_weights('./training/student_distilled.hdf5')
print("🎉 蒸馏完成！已保存为 ./training/student_distilled.hdf5")