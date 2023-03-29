
import torch
import numpy as np
from models import PoetryModel
from torch.autograd import Variable


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datas = np.load("./tang.npz",allow_pickle=True)
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
model=PoetryModel(len(word2ix), embedding_dim=128, hidden_dim=256)
model.load_state_dict(torch.load('model1656255176.064103.pth'))
model.to(DEVICE)#很重要，不要忘记这个！


def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    # 读取唐诗的“头”
    results = []
    start_word_len = len(start_words)

    # 设置第一个词为<START>
    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    input = input.to(DEVICE)
    hidden = None
    index = 0  # 指示已生成了多少句
    pre_word = '<START>'  # 上一个词
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = Variable(input.data.new([word2ix[word]])).view(1, 1)
    # 生成藏头诗
    for i in range(max_gen_len_acrostic):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        # 如果遇到标志一句的结尾，喂入下一个“头”
        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果生成的诗已经包含全部“头”，则结束
            if index == start_word_len:
                break
            # 把“头”作为输入喂入模型
            else:
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        # 否则，把上一次预测作为下一个词输入
        else:
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w

    return results


start_words_acrostic = '风花月雪'  # 唐诗的“头”
max_gen_len_acrostic = 120  # 生成唐诗的最长长度
prefix_words = None
results_acrostic = gen_acrostic(model, start_words_acrostic, ix2word, word2ix, prefix_words)

poetry = ''
for word in results_acrostic:
    poetry += word
    if word == '。' or word == '!':
        poetry += '\n'

print(poetry)
