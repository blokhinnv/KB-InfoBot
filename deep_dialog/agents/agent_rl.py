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

# conda install -c toli lasagne
# pip install lasagne
# pip install Lasagne==0.1
# conda install theano
# F:\Anaconda3\envs\torch_kbinfo\Lib\site-packages\theano\compat -> return x.decode('cp1251')
def categorical_sample(probs: List[float], mode='sample'):
    '''Если mode=max, то возвращает аргмакс
    если sample - то генерирует случайное число x от 0 до 1 и идет по списку, пока не накопит сумму >= x и возвращает
    индекс, при котором это произошло
    '''
    if mode=='max':
        return np.argmax(probs)
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
            learning_rate_rl=0.005, batch_size=32, ment=0.1):
        # 2-layer MLP
        self.in_size = in_size # x and y coordinate
        self.out_size = out_size # up, down, right, left
        self.batch_size = batch_size
        self.learning_rate = learning_rate_rl
        self.n_hid = n_hid

        '''построение модели'''
        # ftensor3: float32, shape=(?,?,?)
        # fvector: float32 shape=(?,)
        # itensor3: int32, shape=(?,?,?)
        # imatrix: int32 shape=(?, ?)
        # ftensor3(name)
        input_var, turn_mask, act_mask, reward_var = T.dtensor3('in'), T.imatrix('tm'), \
                T.itensor3('am'), T.dvector('r')

        # in_var = T.reshape(input_var, (input_var.shape[0] * input_var.shape[1], self.in_size))

        l_mask_in = L.InputLayer(shape=(None,None), input_var=turn_mask)

        pol_in = T.dmatrix('pol-h')

        # l_in = L.InputLayer(shape=(batch_size, max_turns, self.in_size)
        l_in = L.InputLayer(shape=(None, None, self.in_size), input_var=input_var)
        # mask_input: Layer which allows for a sequence mask to be input, for when sequences are of variable length
        l_pol_rnn = L.GRULayer(l_in, n_hid, hid_init=pol_in, mask_input=l_mask_in) # B x H x D
        pol_out = L.get_output(l_pol_rnn)[:,-1,:]
        l_den_in = L.ReshapeLayer(l_pol_rnn, (turn_mask.shape[0] * turn_mask.shape[1], n_hid)) # BH x D
        l_out = L.DenseLayer(l_den_in, self.out_size, nonlinearity=lasagne.nonlinearities.softmax)

        self.network = l_out
        self.params = L.get_all_params(self.network)

        # rl
        probs = L.get_output(self.network) # BH x A
        out_probs = T.reshape(probs, (input_var.shape[0], input_var.shape[1], self.out_size)) # B x H x A
        log_probs = T.log(out_probs)

        # act_probs [batch x max_turn] act_probs[b][i] - значение логарифма вероятности выбранного на шаге i действия
        act_probs = (log_probs * act_mask).sum(axis=2) # B x H

        # * turn_mask - маска для сделанных действий в этом эпизоде
        # ep_probs[b] - сумма логарифмов вероятностей действий, сделанных в эпизоде b
        ep_probs = (act_probs * turn_mask).sum(axis=1) # B

        # энтропия для каждого эпизода
        H_probs = -T.sum(T.sum(out_probs * log_probs, axis=2), axis=1) # B

        # reward_var - награды за эпизоды
        self.loss = 0. - T.mean(ep_probs * reward_var + ment * H_probs)

        updates = lasagne.updates.rmsprop(self.loss, self.params, learning_rate=learning_rate_rl, \
                epsilon=1e-4)

        self.inps = [input_var, turn_mask, act_mask, reward_var, pol_in]
        # obj_fn = train_fn - updates (для оценки)
        self.train_fn = theano.function(self.inps, self.loss, updates=updates)
        self.obj_fn = theano.function(self.inps, self.loss)
        self.act_fn = theano.function([input_var, turn_mask, pol_in], [out_probs, pol_out])

        # sl
        
        sl_loss = 0. - T.mean(ep_probs)
        sl_updates = lasagne.updates.rmsprop(sl_loss, self.params, learning_rate=learning_rate_sl, \
                epsilon=1e-4)

        self.sl_train_fn = theano.function([input_var, turn_mask, act_mask, pol_in], sl_loss, updates=sl_updates)
        self.sl_obj_fn = theano.function([input_var, turn_mask, act_mask, pol_in], sl_loss)

    '''блок функций train/evalute RL/SL'''
    def train(self, inp, tur, act, rew, pin):
        return self.train_fn(inp, tur, act, rew, pin)

    def evaluate(self, inp, tur, act, rew, pin):
        return self.obj_fn(inp, tur, act, rew, pin)

    def sl_train(self, inp, tur, act, pin):
        return self.sl_train_fn(inp, tur, act, pin)

    def sl_evaluate(self, inp, tur, act, pin):
        return self.sl_obj_fn(inp, tur, act, pin)
    '''конец блока функций train/evalute RL/SL'''

    def act(self, inp: List[float], pin:List, mode='sample'):
        '''
        При вызове из agent.next:
            inp := p_vector длинный тензор (1, 1, in_size) со всей информацией по состоянию
            pin := state['pol_state'] - матрица 1 х self.n_hid (сначала - нули)
        '''
        tur = np.ones((inp.shape[0], inp.shape[1])).astype('int8')
        # act_p - model.out_probs(inp.shape[0] x inp.shape[1] x n_out) - вер-ти классов
        # p_out - model.pol_out(inp.shape[0] x n_hid) - последнее скрытое состояние RNN
        act_p, p_out = self.act_fn(inp, tur, pin)
        return categorical_sample(act_p.flatten(), mode=mode), act_p.flatten(), p_out

    def anneal_lr(self):
        '''Режет скорость обучения модели вдвое'''
        self.learning_rate /= 2.
        updates = lasagne.updates.rmsprop(self.loss, self.params, learning_rate=self.learning_rate, \
                epsilon=1e-4)
        self.train_fn = theano.function(self.inps, self.loss, updates=updates)

    # def _debug(self, inp, tur, act, rew):
    #     print('Input = {}, Action = {}, Reward = {}'.format(inp, act, rew))
    #     out = self.debug_fn(inp, tur, act, rew)
    #     for item in out:
    #         print(item)

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
        i = [self.input_pool[ii] for ii in index]
        a = [self.actmask_pool[ii] for ii in index]
        r = [self.reward_pool[ii] for ii in index]
        t = [self.turnmask_pool[ii] for ii in index]
        return np.asarray(i, dtype='float32'), \
                np.asarray(t, dtype='int32'), \
                np.asarray(a, dtype='int32'), \
                np.asarray(r, dtype='float32')

    def update(self, verbose=False, regime='RL'):
        '''Шаг обучения'''
        i, t, a, r = self._get_minibatch(self.batch_size)
        pi = np.zeros((self.batch_size, self.n_hid)).astype('float32')
        # np.repeat(z[np.newaxis,:], 2, axis=0)
        if verbose: 
            print(i, t, a, r)
        if regime=='RL':
            r -= np.mean(r)
            g = self.train(i, t, a, r, pi)
        else:
            g = self.sl_train(i, t, a, pi)
        return g

    def eval_objective(self, N):
        '''Ошибка в коде, надеюсь, что нигде не вызывается'''
        try:
            obj = self.evaluate(self.eval_i, self.eval_t, self.eval_a, self.eval_r)
        except AttributeError:
            self.eval_i, self.eval_t, self.eval_a, self.eval_r = self._get_minibatch(N)
            obj = self.evaluate(self.eval_i, self.eval_t, self.eval_a, self.eval_r)
        return obj

    '''Методы для сохранения/загрузки модели'''
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            data = pkl.load(f)
        L.set_all_param_values(self.network, data)

    def save_model(self, save_path):
        data = L.get_all_param_values(self.network)
        with open(save_path, 'wb') as f:
            pkl.dump(data, f)
            
