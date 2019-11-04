import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

from generate_data import load_json, phrase_path, pinyin_path, zi_path

py_list = load_json(pinyin_path)
zi_list = load_json(zi_path)
py_list.insert(0, '_')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pinyin2id(pys):
    res = []
    for py in pys:
        try:
            res.append(py_list.index(py))
        except Exception:
            res.append(0)
    return res

def hanzi2id(words):
    res = []
    for zi in words:
        try:
            res.append(zi_list.index(zi))
        except Exception:
            res.append(0)
    return res

def one_hot(word):
    res = torch.zeros(len(word), len(zi_list)).long()
    for i, zi in enumerate(word):
        res[i][zi_list.index(zi)] = 1
    return res

class MyDataset(Dataset):
    def __init__(self):
        data_dict = load_json(phrase_path)
        self.data_list = [(key, pinyin2id((value.split(',')[0].split()))) for key, value in data_dict.items()]

    def __getitem__(self, index):
        return self.data_list[index][0], self.data_list[index][1]

    def __len__(self):
        return len(self.data_list)

def gen_batch(data):
    words, labels = zip(*data)
    words_len = [len(word) for word in words]
    # words_pad = torch.zeros((len(words), max(words_len), len(zi_list))).long()
    words_pad = torch.zeros((len(words), max(words_len))).long()
    labels_pad = torch.zeros((len(words), max(words_len))).long()
    for i, word in enumerate(words):
        # words_pad[i, :len(word)] = one_hot(word)
        words_pad[i, :len(word)] = torch.Tensor(hanzi2id(word))

    
    for i, label in enumerate(labels):
        labels_pad[i, :len(label)] = torch.Tensor(label)
    
    return words_pad, labels_pad

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.emb = nn.Embedding(len(zi_list), 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, len(py_list))
        self.ac = torch.relu

    def forward(self, x):
        x = self.emb(x)
        x = self.drop(x)
        h1 = self.drop(self.ac(self.fc1(x)))
        # h1 = self.bn2(h1)
        h2 = self.drop(self.ac(self.fc2(h1)))
        # h2 = self.bn3(h2)
        h3 = self.drop(self.ac(self.fc3(h2)))
        out = F.softmax(h3, -1)
        return out


def train(model, optimizer, dataloader):
    model.train()
    epochs = 10
    loss_func = F.cross_entropy
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for step, (words, labels) in enumerate(myDataLoader):
            words, labels = words.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(words)
            loss = torch.Tensor([0]).to(DEVICE)
            for i in range(out.shape[0]):
                loss += loss_func(out[i], labels[i])
            loss.backward()
            optimizer.step()
            if step % 30 == 0:
                print("loss %.3f" % loss.item())

if __name__ == '__main__':
    # words = ['你好吗', '吃饭了吗']
    # labels = [[1, 3, 5], [9, 8, 123, 7]]
    # datas = zip(words, labels)
    # w_pad, l_pad = gen_batch(datas)
    # model = MyModel()
    # optimizer = optim.Adam(model.parameters())
    # loss_func = nn.CrossEntropyLoss()
    # optimizer.zero_grad()
    # out = model(w_pad)
    # loss = torch.Tensor([0])
    # for i in range(out.shape[0]):
    #     loss += loss_func(out[i], l_pad[i])
    # # out = torch.argmax(out, dim=-1)
    # # loss = loss_func(out.float(), l_pad)
    # loss.backward()
    # optimizer.step()
    # pass

    dataset = MyDataset()
    myDataLoader = DataLoader(dataset, batch_size=16, collate_fn=gen_batch, shuffle=True)

    model = MyModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    train(model, optimizer, myDataLoader)
    
