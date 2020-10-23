import torch
import torch.nn as nn
from torch import optim
import numpy as np

class TreeRecursiveNN(nn.Module):
    def __init__(self, embed_dict, embed_size, hidden_size, use_relations):
        super(TreeRecursiveNN, self).__init__()
        self.embed_dict = embed_dict
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embeddings = self.init_embeddings()
        self.use_relations = use_relations
        self.use_embeds = True

        self.Wforget = nn.Linear(embed_size, hidden_size, bias=True)
        self.Uforget_l_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uforget_l_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uforget_r_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uforget_r_r = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Winput = nn.Linear(embed_size, hidden_size, bias=True)
        self.Uinput_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uinput_r = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Woutput = nn.Linear(embed_size, hidden_size, bias=True)
        self.Uoutput_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uoutput_r = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wupdate = nn.Linear(embed_size, hidden_size, bias=True)
        self.Uupdate_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uupdate_r = nn.Linear(hidden_size, hidden_size, bias=False)

        self.tree2scores = nn.Linear(hidden_size * 2, 3, bias=True)

    def forward(self, input_tree):
        root_hidden_output = self.forward_recurse(input_tree)
        return self.tree2scores(root_hidden_output)[0], root_hidden_output

    def forward_recurse(self, input_tree):
        if (input_tree.left_child is None):
            l_child_hidden_state = self.init_hidden()
            l_child_cell = self.init_hidden()
            r_child_hidden_state = self.init_hidden()
            r_child_cell = self.init_hidden()
        else:
            l_child_hidden_state, l_child_cell = self.forward_recurse(input_tree.left_child)
            r_child_hidden_state, r_child_cell = self.forward_recurse(input_tree.right_child)
        # Embedding for the current discourse role (node)
        mononuclear = ["Joint", "Contrast", "TextualOrganization", "Same-Unit"]
        if (input_tree.role == 'Root'):
            return torch.cat((l_child_hidden_state, r_child_hidden_state), 2)
        if self.use_embeds:
            if (input_tree.rel_type in mononuclear and self.use_relations):
                if self.use_relations:
                    root_embedding = self.embeddings(self.embed_dict[input_tree.rel_type])
                else:
                    root_embedding = self.embeddings(self.embed_dict['Nucleus'])
            else:
                if self.use_relations:
                    root_embedding = self.embeddings(self.embed_dict[input_tree.rel_type + "_" + input_tree.role])
                else:
                    root_embedding = self.embeddings(self.embed_dict[input_tree.role])
        else:
            root_embedding = self.init_root_embedding()
        # RNN gates
        forget_gate_left = torch.sigmoid(self.Wforget(root_embedding) + self.Uforget_l_l(l_child_hidden_state) + self.Uforget_l_r(r_child_hidden_state))
        forget_gate_right = torch.sigmoid(self.Wforget(root_embedding) + self.Uforget_r_l(l_child_hidden_state)
                               + self.Uforget_r_r(r_child_hidden_state))
        input_gate = torch.sigmoid(self.Winput(root_embedding) + self.Uinput_l(l_child_hidden_state)
                               + self.Uinput_r(r_child_hidden_state))
        output_gate = torch.sigmoid(self.Woutput(root_embedding) + self.Uoutput_l(l_child_hidden_state)
                               + self.Uoutput_r(r_child_hidden_state))
        update_gate = torch.tanh(self.Wupdate(root_embedding) + self.Uupdate_l(l_child_hidden_state)
                               + self.Uupdate_r(r_child_hidden_state))
        cell = input_gate * update_gate + forget_gate_left * l_child_cell + forget_gate_right * r_child_cell
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell

    def init_embeddings(self):
        return nn.Embedding(len(self.embed_dict), self.embed_size)
    
    def init_root_embedding(self):
        return torch.zeros(1, self.embed_size, device="cpu")

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device="cpu")
