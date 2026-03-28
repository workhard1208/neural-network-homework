# neural-network-homework
## 第一次作业


### 文件说明

- `homework1.ipynb`：完整实验报告（Jupyter Notebook）
- `homework1.pdf`：完整实验报告（PDF）
- `main.py`：完整代码
- `Concrete_Data_Yeh.csv`：数据集

### 方法

- **特征分析**：皮尔逊相关系数分析各特征与强度的相关性
- **线性回归**：梯度下降 + 动量 + Adagrad自适应学习率
- **神经网络**：4层全连接网络，ReLU激活，Adam优化器，PyTorch实现


### 结果

| 方法 | 测试集MSE | 测试集RMSE |
|------|---------|---------|
| 线性回归 + 动量 + Adagrad | 95.9709 | 9.7965 MPa |
| 神经网络 | 25.3289 | 5.0328 MPa |
