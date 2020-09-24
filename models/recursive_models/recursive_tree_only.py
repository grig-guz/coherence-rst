import torch
import torch.nn as nn
from torch import optim
import numpy as np

class TreeRecursiveTreeOnlyNN(nn.Module):
    def __init__(self, hidden_size):
        super(TreeRecursiveTreeOnlyNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.Uforget_l_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uforget_l_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uforget_r_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uforget_r_r = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Uinput_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uinput_r = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Uoutput_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uoutput_r = nn.Linear(hidden_size, hidden_size, bias=False)

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
        if (input_tree.role == 'Root'):
            return torch.cat((l_child_hidden_state, r_child_hidden_state), 2)
        # RNN gates
        forget_gate_left = torch.sigmoid(self.Uforget_l_l(l_child_hidden_state) + self.Uforget_l_r(r_child_hidden_state))
        forget_gate_right = torch.sigmoid(self.Uforget_r_l(l_child_hidden_state)
                               + self.Uforget_r_r(r_child_hidden_state))
        input_gate = torch.sigmoid(self.Uinput_l(l_child_hidden_state)
                               + self.Uinput_r(r_child_hidden_state))
        output_gate = torch.sigmoid(self.Uoutput_l(l_child_hidden_state)
                               + self.Uoutput_r(r_child_hidden_state))
        update_gate = torch.tanh(self.Uupdate_l(l_child_hidden_state)
                               + self.Uupdate_r(r_child_hidden_state))
        cell = input_gate * update_gate + forget_gate_left * l_child_cell + forget_gate_right * r_child_cell
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device="cpu", requires_grad = True)