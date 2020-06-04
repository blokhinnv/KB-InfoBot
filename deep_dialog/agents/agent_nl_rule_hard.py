from deep_dialog import dialog_config, tools
from deep_dialog.agents.agent import Agent
from deep_dialog.agents.hardDB import HardDB
from deep_dialog.agents.belief_tracker import BeliefTracker
from deep_dialog.agents.utils import *

from collections import Counter, defaultdict

import operator
import random
import math
import numpy as np
import pickle as pkl
import copy
import re
import nltk

class AgentNLRuleHard(Agent,HardDB,BeliefTracker):

    def initialize_episode(self):
        '''Собираем словарь state
        из state.database - база на входе - удаляем столбцы, которые не в dialog_config.inform_slots (таких нет)
        инициализирует self.state['inform_slots'] beliefs частотностями значений слотов
        В agent_act_rule inform_slots был словарь слот: значение, тут слот: частотности возможных значений!
        '''
        self.state = {}
        self.state['database'] = pkl.loads(pkl.dumps(self.database,-1))

        for slot in self.state['database'].slots:
            if slot not in dialog_config.inform_slots: 
                self.state['database'].delete_slot(slot)

        self.state['prevact'] = 'begin@begin'
        self.state['inform_slots'] = self._init_beliefs()
        self.state['turn'] = 0
        # p_db_i - (вектор длины N из 1/N)
        p_db_i = np.ones((self.state['database'].N,) ) / self.state['database'].N

        # .state['inform_slots'].keys => названия слотов
        # p_db_i = p_T^t(i)
        # .state['database'].priors => p_j^0(v)
        # self.state['init_entropy'] := H(p_j^0)
        self.state['init_entropy'] = calc_entropies(self.state['inform_slots'], p_db_i, self.state['database'])
        self.state['num_requests'] = {s:0 for s in self.state['inform_slots'].keys()}
        self.state['slot_tracker'] = set()
        self.state['dont_care'] = set()

    ''' get next action based on rules '''
    def next(self, user_action, verbose=False):
        '''
        Пересчитывает состояние self.state['inform_slots'] по текущему высказыванию
        Проводит жесткий поиск по БД
        Нормирует измененный self.state['inform_slots'] до 1 и считает энтропию столбцов
        Если ничего не нашлось - запрашивает слот
        Если нашлось 1 сущность - возвращает
        Если нашлось несколько - запрашивает слот с максимальной энтропией (а в статье написано - с минимальной!)
        при выполнении условий из раздела @4.4
        '''

        # self._update_state из belief_tracker
        # меняет значения (сначала - частотности) в self.state['inform_slots']
        self._update_state(user_action['nl_sentence'], upd=self.upd, verbose=verbose)
        self.state['turn'] += 1

        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        # self._check_db() - жесткий поиск по БД
        db_status, db_index = self._check_db()

        H_slots = {}
        for s in dialog_config.inform_slots:
            s_p = self.state['inform_slots'][s] / self.state['inform_slots'][s].sum()
            H_slots[s] = tools.entropy_p(s_p)

        sorted_entropies = sorted(H_slots.items(), key=operator.itemgetter(1), reverse=True)
        if verbose:
            print('Agent slot belief entropies - ')
            print(' '.join(['%s:%.2f' %(k,v) for k,v in H_slots.items()]))

        if not db_status:
            # no match, some error, re-ask some slot
            act['diaact'] = 'request'
            request_slot = random.choice(self.state['inform_slots'].keys())
            act['request_slots'][request_slot] = 'UNK'
            self.state['prevact'] = 'request@%s' %request_slot
            self.state['num_requests'][request_slot] += 1
        elif len(db_status)==1:
            print("Informing!")
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_index)
            self.state['prevact'] = 'inform@inform'
        else:
            req = False
            for (s, h) in sorted_entropies:
                '''
                    @4.4 Hand Crafted Policy
                    self.ts := alpha_r
                    self.frac := beta
                    self.max_req := Q
                '''
                if (H_slots[s] < self.frac * self.state['init_entropy'][s] or 
                    H_slots[s] < self.ts or
                    self.state['num_requests'][s] >= self.max_req):
                    continue
                act['diaact'] = 'request'
                act['request_slots'][s] = 'UNK'
                self.state['prevact'] = 'request@%s' % s
                self.state['num_requests'][s] += 1
                req = True
                break
            if not req:
                # agent confident about all slots, inform
                act['diaact'] = 'inform'
                act['target'] = self._inform(db_index)
                self.state['prevact'] = 'inform@inform'

        act['posterior'] = np.zeros((len(self.database.labels),))
        act['posterior'][db_index] = 1./len(db_index)

        return act

    def terminate_episode(self, user_action):
        return
