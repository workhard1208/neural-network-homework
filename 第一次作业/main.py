import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# ── 读取数据 ──
df = pd.read_csv('Concrete_Data_Yeh.csv')
print("数据大小：", df.shape)
print("\n前5行数据预览：")
print(df.head())
print(df.describe())

# ── 相关性分析 ──
correlations = df.corr()['csMPa'].drop('csMPa')
correlations_abs = correlations.abs().sort_values(ascending=False)
print("\n各特征与混凝土强度的相关性：")
print(correlations_abs)

plt.figure(figsize=(10, 5))
correlations_abs.plot(kind='bar', color='steelblue')
plt.title("各特征与混凝土强度的相关性")
plt.ylabel("相关系数（绝对值）")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("01_correlation.png", dpi=150)
plt.close()
print("图已保存：01_correlation.png")

# ── 数据准备：随机打乱划分 ──
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print("\n训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)

# ── 标准化 ──
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("标准化后训练集均值（应接近0）：", X_train_scaled.mean(axis=0).round(2))
print("标准化后训练集标准差（应接近1）：", X_train_scaled.std(axis=0).round(2))

# ════════════════════════════════════════════
#   方法一：线性回归 + 动量 + Adagrad
# ════════════════════════════════════════════
print("\n── 线性回归 + 动量 + Adagrad ──")

np.random.seed(42)
w = np.random.randn(8) * 0.01  # 8个权重，随机初始化
b = 0.0                         # 偏置初始为0

lr       = 0.1    # 基础学习率（Adagrad需要比普通梯度下降大一些）
epochs   = 5000   # 训练轮数
N        = len(X_train_scaled)
momentum = 0.9    # 动量系数λ：保留上一步方向的比例（课件第32页）
eps      = 1e-8   # 防止除以0

# 动量初始化
m_w = np.zeros(8)  # w的动量，初始为0
m_b = 0.0          # b的动量，初始为0

# Adagrad梯度平方累积，初始为0
G_w = np.zeros(8)
G_b = 0.0

loss_lr = []

for i in range(epochs):
    # 前向传播
    Y_pred = X_train_scaled @ w + b

    # 计算损失
    loss = np.mean((Y_train - Y_pred) ** 2)
    loss_lr.append(loss)

    # 计算梯度
    error  = Y_train - Y_pred
    grad_w = -2/N * (X_train_scaled.T @ error)
    grad_b = -2/N * np.sum(error)

    # Adagrad：累积历史梯度平方（课件第26页）
    # 梯度大的参数学习率自动变小，梯度小的参数学习率相对较大
    G_w += grad_w ** 2
    G_b += grad_b ** 2

    # 自适应学习率
    adaptive_lr_w = lr / (np.sqrt(G_w) + eps)
    adaptive_lr_b = lr / (np.sqrt(G_b) + eps)

    # 动量更新（课件第32页）
    # movement = λ * 上一步movement - 自适应lr * 当前梯度
    m_w = momentum * m_w - adaptive_lr_w * grad_w
    m_b = momentum * m_b - adaptive_lr_b * grad_b

    # 更新参数
    w = w + m_w
    b = b + m_b

Y_pred_test_lr = X_test_scaled @ w + b
mse_lr  = np.mean((Y_test - Y_pred_test_lr) ** 2)
rmse_lr = np.sqrt(mse_lr)
print(f"线性回归 测试集MSE：{mse_lr:.4f}")
print(f"线性回归 测试集RMSE：{rmse_lr:.4f} MPa")

plt.figure(figsize=(8, 4))
plt.plot(loss_lr)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("线性回归（动量+Adagrad）训练Loss曲线")
plt.savefig("02_linear_loss.png", dpi=150)
plt.close()
print("图已保存：02_linear_loss.png")

plt.figure(figsize=(5, 5))
plt.scatter(Y_test, Y_pred_test_lr, alpha=0.6, s=20, color='steelblue')
plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel("真实值 (MPa)")
plt.ylabel("预测值 (MPa)")
plt.title(f"线性回归：预测值 vs 真实值\nMSE={mse_lr:.2f}")
plt.savefig("03_linear_pred.png", dpi=150)
plt.close()
print("图已保存：03_linear_pred.png")

# ════════════════════════════════
#   方法二：神经网络（PyTorch）
# ════════════════════════════════
print("\n── 神经网络 ──")

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(Y_train,        dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_test_t  = torch.tensor(Y_test,         dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_train_t, y_train_t)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

class ConcreteNet(nn.Module):
    def __init__(self):
        super(ConcreteNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)

torch.manual_seed(42)
model = ConcreteNet()
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 学习率调度器：每300轮学习率乘以0.5
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

epochs_nn = 2000  # 增加到2000轮，训练更充分
loss_nn = []

for epoch in range(epochs_nn):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss   = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)
    scheduler.step()
    avg_loss = epoch_loss / len(X_train_t)
    loss_nn.append(avg_loss)
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs_nn}  训练Loss: {avg_loss:.4f}  "
              f"学习率: {scheduler.get_last_lr()[0]:.6f}")

model.eval()
with torch.no_grad():
    y_pred_test_nn = model(X_test_t).numpy().flatten()

mse_nn  = np.mean((Y_test - y_pred_test_nn) ** 2)
rmse_nn = np.sqrt(mse_nn)
print(f"\n神经网络 测试集MSE：{mse_nn:.4f}")
print(f"神经网络 测试集RMSE：{rmse_nn:.4f} MPa")

# ── 最终对比 ──
print("\n" + "="*40)
print(f"{'方法':<15} {'MSE':<12} {'RMSE':<12}")
print("-"*40)
print(f"{'线性回归':<15} {mse_lr:<12.4f} {rmse_lr:<12.4f}")
print(f"{'神经网络':<15} {mse_nn:<12.4f} {rmse_nn:<12.4f}")
print(f"\n神经网络相比线性回归提升：{(mse_lr - mse_nn) / mse_lr * 100:.1f}%")

plt.figure(figsize=(8, 4))
plt.plot(loss_nn)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("神经网络训练Loss曲线")
plt.savefig("04_nn_loss.png", dpi=150)
plt.close()
print("图已保存：04_nn_loss.png")

plt.figure(figsize=(5, 5))
plt.scatter(Y_test, y_pred_test_nn, alpha=0.6, s=20, color='tomato')
plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()], 'k--')
plt.xlabel("真实值 (MPa)")
plt.ylabel("预测值 (MPa)")
plt.title(f"神经网络：预测值 vs 真实值\nMSE={mse_nn:.2f}")
plt.savefig("05_nn_pred.png", dpi=150)
plt.close()
print("图已保存：05_nn_pred.png")

print("\n全部完成！共生成5张图片在当前文件夹。")