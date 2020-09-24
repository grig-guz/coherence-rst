import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
from datasets.parse_codra import *
import os
import random
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pad_sequence, pack_padded_sequence
from nltk.tokenize import word_tokenize, sent_tokenize


class EduDataset(Dataset):

    def __init__(self, txt_folder, dataset_names, glove_dict, device, use_edu=True, is_cnn=False):
        """
            X - array of 2-tuples containing tree roots for coherent and incoherent text.
            y - array of +1s and -1s, For each pair of trees, corresponding y should be +1 if first tree is in right order
            (and second tree is in wrong order) and -1 if the second tree is in right order
        """
        if not txt_folder.endswith("/"):
            txt_folder += "/"
        self.device = device
        self.X = []
        self.y = []
        for dataset_name in dataset_names:
            data_csv = pd.read_csv(txt_folder + dataset_name + "/" +  dataset_name + ".csv")
            count = 0
            for filename in os.listdir(txt_folder + dataset_name):
                if filename.endswith(".txt") and not filename.endswith("tree.txt") and not filename.endswith("depparse.txt"):
#               if filename.endswith(".edu"):
                    raw_text = EduDataset.get_raw_text(txt_folder + dataset_name, filename)
                    #tree_pointer = parse_codra_tree(txt_folder + dataset_name + "/" + filename, raw_text, is_cnn)
                    #if is_cnn:sent_tokenize(text)
                    #    tree_pointer = tree_pointer.make_grid()
                    #edus = list(filter(lambda x: x != "EDU_BREAK" and x != "\n" and x != '', raw_text.replace("EDU_BREAK", "\n").split("\n")))
                    edus = sent_tokenize(raw_text)
                    edus = [word_tokenize(edu) for edu in edus]
                    for edu_id, edu in enumerate(edus):
                        for word_id, word in enumerate(edu):
                            edus[edu_id][word_id] = glove_dict.get(edus[edu_id][word_id].lower(), 0)
                    edus = [torch.LongTensor(edu, device=device) for edu in edus]
                    if (edus == []):
                        continue

                    self.X.append(edus)
                    #id = filename[1:-8]
                    id = filename[1:-4]
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
        X = self.X[idx]
        X = list(filter(lambda x: len(x) > 0, X))
        padded_edus, sort_idx = self.pad_edu_tensors(X)
        #print(sort_idx)
        return (padded_edus, sort_idx), torch.LongTensor(self.y[idx]).to(self.device)

    def get_raw_text(dir, filename):
        rawfile = open(dir + "/" + filename)
        return rawfile.read()

    def pad_edu_tensors(self, document_edus):
        lengths = torch.LongTensor([len(indices) for indices in document_edus]).to(self.device)
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        padded_edu_words = pad_sequence(document_edus, batch_first=True)

        padded_edu_words = padded_edu_words[sorted_idx]
        packed_seq_words = pack_padded_sequence(padded_edu_words, lengths=lengths_sorted.tolist(), batch_first=True)
        return packed_seq_words, sorted_idx
