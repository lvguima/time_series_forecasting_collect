import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # 前向传播LSTM
        out,r = self.lstm(x, (h0.detach(), c0.detach()))

        # 解码LSTM最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 定义函数来创建训练和测试数据集
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i+look_back, :])
    return np.array(dataX), np.array(dataY)

# 加载数据# 将数据变为minibatch的形式进行训练和验证以及测试
data = pd.read_csv('new_data.csv', header=None)

# 标准化数据
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# 分割数据为训练和测试集
train_size = int(len(data) * 0.9)
test_size = len(data) - train_size
train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

# 创建训练和测试数据集
look_back = 20
train_X, train_Y = create_dataset(train_data, look_back)
test_X, test_Y = create_dataset(test_data, look_back)

# 转换为Tensor类型
train_X = torch.from_numpy(train_X).type(torch.Tensor)
train_Y = torch.from_numpy(train_Y).type(torch.Tensor)
test_X = torch.from_numpy(test_X).type(torch.Tensor)
test_Y = torch.from_numpy(test_Y).type(torch.Tensor)

# 定义模型
input_size = 11
hidden_size = 200
output_size = 11
num_layers = 2
model = LSTM(input_size, hidden_size, output_size, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_X)
    loss = criterion(outputs, train_Y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 测试模型
model.eval()
test_pred = model(test_X)
test_pred = test_pred.detach().numpy()

# 反标准化数据
test_Y = scaler.inverse_transform(test_Y.numpy())
test_pred = scaler.inverse_transform(test_pred)


real = pd.DataFrame(test_Y,).to_csv('real1.csv',header=None,index=None)
pred = pd.DataFrame(test_pred).to_csv('pred1.csv',header=None,index=None)


# 计算RMSE误差
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test_Y, test_pred))
print('RMSE: {:.6f}'.format(rmse))
from matplotlib import pyplot as plt
i=9
plt.plot(test_Y[:,i], label='True')
plt.plot(test_pred[:,i], label='Predicted')
plt.legend()
plt.show()