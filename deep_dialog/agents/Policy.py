import torch
import torch.nn as nn
import torch.nn.functional as F

class RLPolicy(nn.Module):
    def __init__(self, in_size, out_size, n_hid=10):
        super(RLPolicy, self).__init__()
        self.in_size = in_size 
        self.out_size = out_size 
        self.n_hid = n_hid
         
        self.gru = nn.GRU(input_size=in_size, hidden_size=n_hid, num_layers=1, batch_first=True)      
        self.fc1 = nn.Linear(in_features=self.n_hid, out_features=out_size)

        
    def forward(self, input_var, pol_in):
        batch_size, seq_len, _ = input_var.shape
        output, pol_out = self.gru(input_var, pol_in)
        # output.shape = (batch_size x seq_len x n_hid)
        # pol_out.shape = (batch_size x 1 x n_hid)
        output = output.reshape((batch_size * seq_len, self.n_hid))
        
        probs = F.softmax(self.fc1(output), dim=1)
        # probs.shape = (batch_size x seq_len, out_size)
        out_probs = probs.view((batch_size, seq_len, self.out_size))
        
        return out_probs, pol_out
    