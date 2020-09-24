import torch
import torch.nn as nn
from torch import optim
import numpy as np
import nltk

class GCDCModel(nn.Module):
    def __init__(self, embed_dict, glove, embed_size, glove_dim, hidden_size):
        super(GCDCModel, self).__init__()
        self.embed_dict = embed_dict
        self.glove = glove
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embeddings = self.init_embeddings()

        self.embed2scores = nn.Linear(hidden_size * 2, 3, bias=True)
        self.dropout = nn.Dropout()
        self.gcdc_2ens = nn.Linear(hidden_size, hidden_size * 2, bias=True)
        self.word_gru = nn.LSTM(300, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.sent_gru = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.par_gru = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, input):
        _, paragraphs = input
        out = self.gcdc_model(paragraphs)
        out = torch.relu(out)
        out = self.dropout(out)
        scores = self.embed2scores(out)
        return scores.unsqueeze(0), out

    def gcdc_model(self, paragraphs):
        par_embeds = []
        for i, paragraph in enumerate(paragraphs):
            sent_embeds = []
            for j, sentence in enumerate(paragraphs[i]):
                if (sentence != []):
                    _, sent_embed = self.word_gru(self.construct_words_embeddings(sentence))
                    # Sum both directions
                    sent_embeds.append(sent_embed[0][0][0] + sent_embed[0][1][0])
            _, par_embed = self.sent_gru(torch.stack(sent_embeds).unsqueeze(0))
            par_embeds.append(par_embed[0][0][0] + par_embed[0][1][0])
        _, out = self.par_gru(torch.stack(par_embeds).unsqueeze(0))
        out = self.dropout(out[0][0][0] + out[0][1][0])
        return self.gcdc_2ens(out)

    def init_embeddings(self):
        return nn.Embedding(len(self.embed_dict), self.embed_size)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device="cpu", requires_grad = True)

    def construct_words_embeddings(self, sentence):
        matrix_len = len(sentence)
        weights_matrix = np.zeros((matrix_len, 300))
        for i, word in enumerate(sentence):
            try:
                weights_matrix[i] = self.glove[word.lower()]
            except KeyError:
                weights_matrix[i] = np.zeros(300)
        return torch.FloatTensor([weights_matrix])
