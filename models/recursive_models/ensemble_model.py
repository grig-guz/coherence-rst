import torch
import torch.nn as nn
from torch import optim
import numpy as np
import nltk

class EnsembleModel(nn.Module):
    def __init__(self, recursive_model, gcdc_model, hidden_size, out_size=3):
        super(EnsembleModel, self).__init__()
        self.recursive_model = recursive_model
        self.gcdc_model = gcdc_model
        self.dropout = nn.Dropout()
        self.tree2embed = nn.Linear(hidden_size * 4, hidden_size * 2, bias=True)
        self.embed2scores = nn.Linear(hidden_size * 2, out_size, bias=True)

    def forward(self, input):
        _, out_disc = self.recursive_model(input[0])
        _, out_sem = self.gcdc_model(input)
        out = torch.cat([out_disc[0][0], out_sem])
        out = self.dropout(out)
        embed = self.tree2embed(out)
        embed = torch.relu(embed)
        embed = self.dropout(embed)
        out = self.embed2scores(embed).unsqueeze(0)
        return out, embed
