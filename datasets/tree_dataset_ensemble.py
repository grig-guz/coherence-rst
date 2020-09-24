import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nltk import tokenize
from datasets.parse_codra import *
from nltk.tokenize import word_tokenize, sent_tokenize

import os
import random
import pandas as pd
import re

class TreeDatasetEnsemble(Dataset):

    def __init__(self, txt_folder, dataset_names):
        """
            X - array of 2-tuples containing tree roots for coherent and incoherent text.
            y - array of +1s and -1s, For each pair of trees, corresponding y should be +1 if first tree is in right order
            (and second tree is in wrong order) and -1 if the second tree is in right order
        """
        if not txt_folder.endswith("/"):
            txt_folder += "/"

        self.X = []
        self.y = []
        for dataset_name in dataset_names:
            data_csv = pd.read_csv(txt_folder + dataset_name + "/" + dataset_name + ".csv")
            count = 0
            for filename in os.listdir(txt_folder + dataset_name):
                if filename.endswith("_tree.txt"):
                    id = filename[1:-9]
                    paragraphs = self.get_paragraphs(txt_folder + dataset_name + "/", filename)
                    tree_pointer = parse_codra_tree(txt_folder + dataset_name + "/" + filename, 
                                                    (txt_folder + dataset_name + "/" + filename).replace("_tree",""), 
                                                    is_cnn=False)
                    self.X.append((tree_pointer, paragraphs))
                    text_pd_entry = data_csv.loc[data_csv["text_id"].astype(str) == id]
                    # Make labels 0 to 2 instead of 1 to 3
                    label = text_pd_entry["labelA"] - 1
                    label = label.iloc[0]
                    target = torch.LongTensor(1)
                    target[0] = int(label)
                    self.y.append(target)
                    count += 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_paragraphs(self, dir, filename):
        rawfile = open(dir + "/" + filename[0:filename.index("_tree.txt")] + ".txt")
        text = rawfile.read()
        paragraphs = text.split("\n\n")
        for i, paragraph in enumerate(paragraphs):
            paragraphs[i] = sent_tokenize(paragraphs[i])
            for j, sentence in enumerate(paragraphs[i]):
                words = word_tokenize(sentence)
                paragraphs[i][j] = words
        return paragraphs
