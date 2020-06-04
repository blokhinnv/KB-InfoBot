import lasagne
import theano
import lasagne.layers as L
import theano.tensor as T
import numpy as np
import sys

from collections import Counter, defaultdict, deque

import random
import pickle as pkl

from typing import List

from .Policy import RLPolicy
from .Loss import RLLoss, SLLoss
import torch
from torch.optim.lr_scheduler import MultiStepLR


def categorical_sample(probs: List[float], mode='sample'):
    '''Если mode=max, то возвращает аргмакс
    если sample - то генерирует случайное число x от 0 до 1 и идет по списку, пока не накопит сумму >= x и возвращает
    индекс, при котором это произошло
    '''
    if mode=='max':
        return torch.argmax(probs).item()
    else:
        x = np.random.uniform()
        s = probs[0]
        i = 0
        while s < x:
            i += 1
            try:
                s += probs[i]
            except IndexError:
                sys.stderr.write('Sample out of Bounds!! Probs = {} Sample = {}'.format(probs, x))
                return i - 1
        return i

def aggregate_rewards(rewards,discount):
    running_add = 0.
    for t in range(1,len(rewards)):
        running_add += rewards[t]*discount**(t-1)
    return running_add




class RLAgent:
    def _init_model(self, in_size, out_size, n_hid=10, learning_rate_sl=0.005, \
                    learning_rate_rl=0.005, batch_size=32, ment=0.1, anneal_milestone=800):

        self.batch_size = batch_size
        self.ment = ment
        self.policy = RLPolicy(in_size=in_size, out_size=out_size, n_hid=n_hid)
        
        self.rl_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate_rl)
        self.sl_optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=learning_rate_sl, eps=1e-4)

        milestones = [anneal_milestone * i for i in range(1, 10)]
        self.rl_scheduler = MultiStepLR(self.rl_optimizer, milestones=milestones, gamma=0.5)
        # self.sl_scheduler = MultiStepLR(self.sl_optimizer, milestones=milestones, gamma=0.5)

    '''блок функций train/evalute RL/SL'''
    def train(self, inp, tur, act, rew, pin):
        # forward 
        out_probs, pol_out = self.policy(inp, pin)
        loss = RLLoss(out_probs, tur, act, rew, self.ment)
        # backwards
        self.rl_optimizer.zero_grad()
        loss.backward()
        self.rl_optimizer.step()
        self.rl_scheduler.step()
        return loss.item()

    def evaluate(self, inp, tur, act, rew, pin):
        with torch.no_grad():
            out_probs, pol_out = self.policy(inp, pin)
            loss = RLLoss(out_probs, tur, act, rew, self.ment)
            return loss.item()

    def sl_train(self, inp, tur, act, pin):
        # forward 
        out_probs, pol_out = self.policy(inp, pin)
        loss = SLLoss(out_probs, tur, act)
        # backwards
        self.sl_optimizer.zero_grad()
        loss.backward()
        self.sl_optimizer.step()
        return loss.item()

    def sl_evaluate(self, inp, tur, act, pin):
        with torch.no_grad():
            out_probs, pol_out = self.policy(inp, pin)
            loss = SLLoss(out_probs, tur, act)
            return loss.item()
    '''конец блока функций train/evalute RL/SL'''

    def act(self, inp: List[float], pin:List, mode='sample'):
        '''
        При вызове из agent.next:
            inp := p_vector длинный тензор (1, 1, in_size) со всей информацией по состоянию
            pin := state['pol_state'] - матрица 1 х self.n_hid (сначала - нули)
        '''
        inp = torch.Tensor(inp)
        pin = torch.Tensor(pin)
        tur = torch.ones((inp.shape[0], inp.shape[1])).type(torch.int8)
        # act_p - model.out_probs(inp.shape[0] x inp.shape[1] x n_out) - вер-ти классов
        # p_out - model.pol_out(inp.shape[0] x n_hid) - последнее скрытое состояние RNN
        act_p, p_out = self.policy(inp, pin)
        return categorical_sample(act_p.flatten(), mode=mode), act_p.flatten(), p_out

    def anneal_lr(self):
        '''Режет скорость обучения модели вдвое'''
        # self.rl_scheduler.step()

    def _init_experience_pool(self, pool: int) -> None:
        '''pool - размер пула (== batch size)'''
        self.input_pool = deque([], pool)
        self.actmask_pool = deque([], pool)
        self.reward_pool = deque([], pool)
        self.turnmask_pool = deque([], pool)

    def add_to_pool(self, inp, turn, act, rew):
        self.input_pool.append(inp)
        self.actmask_pool.append(act)
        self.reward_pool.append(rew)
        self.turnmask_pool.append(turn)

    def _get_minibatch(self, N):
        '''
        Пулы - очереди с ограничением, сначала пустые
        '''
        n = min(N, len(self.input_pool))
        index = random.sample(range(len(self.input_pool)), n)
        i = torch.Tensor([self.input_pool[ii] for ii in index])
        a = torch.Tensor([self.actmask_pool[ii] for ii in index])
        r = torch.Tensor([self.reward_pool[ii] for ii in index])
        t = torch.Tensor([self.turnmask_pool[ii] for ii in index])
        return i, t, a, r

    def update(self, verbose=False, regime='RL'):
        '''Шаг обучения'''
        i, t, a, r = self._get_minibatch(self.batch_size)
        pi = torch.zeros((1, self.batch_size, self.policy.n_hid))
        # np.repeat(z[np.newaxis,:], 2, axis=0)
        if verbose: 
            print(i, t, a, r)
        if regime=='RL':
            r -= torch.mean(r)
            g = self.train(i, t, a, r, pi)
        else:
            g = self.sl_train(i, t, a, pi)
        return g


    '''Методы для сохранения/загрузки модели'''
    def load_model(self, load_path):
        self.policy.load_state_dict(torch.load(load_path))

    def save_model(self, save_path):
        torch.save(self.policy.state_dict(), save_path)
            
