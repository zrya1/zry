


import torch.nn as nn



class PoetryModel(nn.Module):#父模型是module
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()#继承父亲属性
        self.hidden_dim = hidden_dim#隐藏层大小
        self.embedding = nn.Embedding(vocab_size, embedding_dim)#对数据进行嵌入，到embedding大小的向量上。
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3)#三层的LSTM层
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, vocab_size)
        )
#全连接层和relu层的组合，将降维得到的字词信息进行还原，类似于一个自编码器
    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()

        if hidden is None:#对h_0,c_0初始化
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embedding(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.classifier(output.view(seq_len * batch_size, -1))
#将输入经过lstm网络
        return output, hidden