import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, hidden_size, sample_len, batch_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.sample_len = sample_len
        self.batch_size = batch_size

        self.inp = nn.Linear(sample_len, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = nn.Linear(hidden_size, 16)

    def forward(self, inputs, hidden=None):
        outputs = self.inp(inputs)
        outputs, hidden = self.rnn(outputs, hidden)
        print(outputs.shape)
        outputs = self.out(outputs)
        
        return outputs