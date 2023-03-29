
import numpy as np
import torch
from torch.autograd import Variable
from models import PoetryModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datas = np.load("./tang.npz",allow_pickle=True)
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
model=PoetryModel(len(word2ix), embedding_dim=128, hidden_dim=256)
model.load_state_dict(torch.load('model1656255176.064103.pth'))
model.to(DEVICE)#很重要，不要忘记这个！
#加载模型，并将预训练好的模型参数存到model里

def generate(model, start_words, ix2word, word2ix, max_gen_len, prefix_words=None):
    # 读取唐诗的第一句
    results = list(start_words)
    start_word_len = len(start_words)
    # 设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(DEVICE)
    hidden = None
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = Variable(input.data.new([word2ix[word]])).view(1, 1)
    # 生成唐诗
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        # 读取第一句
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1).to(DEVICE)
        # 生成后面的句子
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1).to(DEVICE)
        # 结束标志
        if w == '<EOP>':
            del results[-1]
            break
    return results


def que():
    start_words = '北塔零空虚'  # 唐诗的第一句
    max_gen_len = 120  # 生成唐诗的最长长度

    prefix_words = None
    results = generate(model, start_words, ix2word, word2ix, max_gen_len, prefix_words)
    poetry = ''
    for word in results:
        poetry += word
        if word == '。' or word == '!':
            poetry += '\n'

    print(poetry)


que()