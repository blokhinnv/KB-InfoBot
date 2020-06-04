from deep_dialog import dialog_config, tools
from deep_dialog.agents.agent import Agent
from deep_dialog.agents.softDB import SoftDB
from deep_dialog.agents.belief_tracker import BeliefTracker
from deep_dialog.agents.utils import *

from collections import Counter, defaultdict

import operator
import random
import math
import numpy as np
import copy
import re
import nltk
import pickle as pkl

class AgentNLRuleSoft(Agent,SoftDB,BeliefTracker):

    def initialize_episode(self) -> None:
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
        p_db_i = (1./self.state['database'].N)*np.ones((self.state['database'].N,))

        # .state['inform_slots'].keys => названия слотов
        # p_db_i = p_T^t(i)
        # .state['database'].priors => p_j^0(v)
        # self.state['init_entropy'] := H(p_j^0)
        self.state['init_entropy'] = calc_entropies(self.state['inform_slots'], p_db_i, 
                self.state['database'])

        self.state['num_requests'] = {s:0 for s in self.state['inform_slots'].keys()}
        self.state['slot_tracker'] = set()
        self.state['dont_care'] = set()

    ''' get next action based on rules '''
    def next(self, user_action, verbose=False):
        '''
        Пересчитывает состояние self.state['inform_slots'] по текущему высказыванию
        Проводит мягкий поиск по БД и возвращает вектор db_probs вероятностей для строк
        Считает entropy statistic (8) по p_T^t(i) := db_probs
        Если она меньше порога (гиперпараметр) - возвращает ответ
        Иначе запрашивает слот с максимальной энтропией (а в статье написано - с минимальной!)
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


        db_probs = self._check_db()
        H_db = tools.entropy_p(db_probs)
        H_slots = calc_entropies(self.state['inform_slots'], db_probs, self.state['database'])
        if verbose:
            print('Agent DB entropy = ', H_db)
            print('Agent slot belief entropies - ')
            print(' '.join(['%s:%.2f' %(k,v) for k,v in H_slots.items()]))

        if H_db < self.tr:
            # agent reasonable confident, inform
            print("Has reasonable confident: ", H_db)
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_probs)  # для softdb - вероятности, сорт. по убыв.
        else:
            sorted_entropies = sorted(H_slots.items(), key=operator.itemgetter(1), reverse=True)
            req = False
            '''
                    @4.4 Hand Crafted Policy
                    self.ts := alpha_r
                    self.frac := beta
                    self.max_req := Q
            '''
            for (s,h) in sorted_entropies:
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
                print('Agent confident about all slots, inform')
                act['diaact'] = 'inform'
                act['target'] = self._inform(db_probs)  # для softdb - вероятности, сорт. по убыв.
                self.state['prevact'] = 'inform@inform'

        act['probs'] = [np.concatenate([self.state['inform_slots'][s] / self.state['inform_slots'][s].sum(), 
                                        np.asarray([float(self.state['database'].inv_counts[s][-1]) / self.state['database'].N])
                                       ])
                        for s in dialog_config.inform_slots]
        act['phis'] = [1. if s in self.state['dont_care'] else 0. for s in dialog_config.inform_slots]
        act['posterior'] = db_probs
        return act

    def terminate_episode(self, user_action):
        return
