import torch
import torch.nn as nn
import torch.nn.functional as F

def RLLoss(out_probs, turn_mask, act_mask, reward_var, ment):
    log_probs = torch.log(out_probs)
    # log_probs.shape = (batch_size, seq_len, out_size)

    # act_probs [batch x max_turn] act_probs[b][i] - значение логарифма вероятности выбранного на шаге i действия
    act_probs = (log_probs * act_mask).sum(dim=2) # B x H

    # * turn_mask - маска для сделанных действий в этом эпизоде
    # ep_probs[b] - сумма логарифмов вероятностей действий, сделанных в эпизоде b
    ep_probs = (act_probs * turn_mask).sum(dim=1) # B

    # энтропия для каждого эпизода
    H_probs = - torch.sum(torch.sum(out_probs * log_probs, dim=2), dim=1) # B

    # reward_var - награды за эпизоды
    loss = 0. - torch.mean(ep_probs * reward_var + ment * H_probs)
    
    return loss

def SLLoss(out_probs, turn_mask, act_mask):
    log_probs = torch.log(out_probs)
    # log_probs.shape = (batch_size, seq_len, out_size)

    # act_probs [batch x max_turn] act_probs[b][i] - значение логарифма вероятности выбранного на шаге i действия
    act_probs = (log_probs * act_mask).sum(dim=2) # B x H

    # * turn_mask - маска для сделанных действий в этом эпизоде
    # ep_probs[b] - сумма логарифмов вероятностей действий, сделанных в эпизоде b
    ep_probs = (act_probs * turn_mask).sum(dim=1) # B

    sl_loss = 0. - torch.mean(ep_probs)
    
    return sl_loss