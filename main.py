# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import torch
import torch.nn as nn
import time
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from models import PoetryModel

BATCH_SIZE = 16
EPOCHS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读入预处理的数据
datas = np.load("./tang.npz",allow_pickle=True)
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
#上述三行，第一行提取的是文件中的data信息，第二行提取的是编码转文字信息，第三行提取的是文字转编码信息

# 转为torch.Tensor,将数据封装为按照batch大小一组的数据
data = torch.from_numpy(data)
train_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)





# 配置模型，是否继续上一次的训练
model = PoetryModel(len(word2ix), embedding_dim=128, hidden_dim=256)
model_path = ''  # 预训练模型路径
if model_path:
    model.load_state_dict(torch.load(model_path))
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=10)#修正模型学习速度


def train(model, dataloader, ix2word, word2ix, device, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0.0

    for batch_idx, data in enumerate(dataloader):
        data = data.long().transpose(1, 0).contiguous()#拷贝并转置数据
        data = data.to(device)
        optimizer.zero_grad()#初始化
        input, target = data[:-1, :], data[1:, :]
        output, _ = model(input)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if (batch_idx + 1) % 200 == 0:
            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data[1]), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))

    train_loss *= BATCH_SIZE
    train_loss /= len(train_loader.dataset)
    print('\ntrain epoch: {}\t average loss: {:.6f}\n'.format(epoch, train_loss))
    scheduler.step()

    return train_loss


train_losses = []

for epoch in range(1, EPOCHS + 1):
    tr_loss = train(model, train_loader, ix2word, word2ix, DEVICE, optimizer, scheduler, epoch)
    train_losses.append(tr_loss)

# 保存模型
filename = "model" + str(time.time()) + ".pth"
torch.save(model.state_dict(), filename)





