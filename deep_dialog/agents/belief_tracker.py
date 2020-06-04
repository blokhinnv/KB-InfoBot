'''
'''

import nltk
import numpy as np
import time
from typing import List

from collections import Counter, defaultdict
from deep_dialog.tools import to_tokens


UPD = 10

class BeliefTracker:
    '''@4.2 Hand-Crafted Tracker'''
    def _search(self, slot_tokens:List, utt_tokens:List):
        '''utt_tokens - список токенов в utt, slot_tokens - список токенов в названии слота'''
        # доля токенов из названия слота, которые встретились в utt
        return float(sum([ww in utt_tokens for ww in slot_tokens]))/len(slot_tokens)

    def _search_slots(self, utt_tokens):
        '''returns: словарь слот: доля токенов из названия слота, которые встретились в utt (если доля >0)'''
        # expr.(4)
        matches = {}
        for slot, slot_tokens in self.state['database'].slot_tokens.items():
            m = self._search(slot_tokens, utt_tokens)
            if m>0.: 
                matches[slot] = m
        return matches

    def _search_values(self, utt_tokens):
        '''returns: словарь слот: {инд значения: доля токенов значения, которые есть в токенах utt}    '''
        matches = {}

        for slot in self.state['database'].slots:  # названия слотов
            matches[slot] = defaultdict(float)
            # находим среди возможных значений слота slot такие, в которых есть токены из utt
            for ss in utt_tokens:
                # self.movie_dict.tokens[slot] - словарь токен: индексы значений
                if ss in self.movie_dict.tokens[slot]:  # если токен из utt в этом словаре
                    for vi in self.movie_dict.tokens[slot][ss]: # по индексам значений
                        matches[slot][vi] += 1. 
            # matches[slot]: инд значения: количество пересечений токенов значения и токенов utt

            # expr.(4)
            for vi, f in matches[slot].items():
                val = self.movie_dict.dict[slot][vi]
                matches[slot][vi] = f / len(nltk.word_tokenize(val))
            # matches[slot]: {инд значения: доля токенов значения val, которые есть в токенах utt }   
        return matches

    ''' update agent state '''
    def _update_state(self, user_utterance, upd=UPD, verbose=False):
        '''Обновляет состояние агента на основании высказывания пользователя
        Сначала ищет слоты и значения в высказывании, потом смотрит
            если какой-то слот запрашивался в вопросе, получили ответ, но не нашелся - больше его не спрашиваем
            если спрашивался и нашлись какие-то похожие значения
                обновляем вероятность каждого из них по правилу (5)
        '''
        # в начале - begin, begin
        prev_act, prev_slot = self.state['prevact'].split('@')
        if verbose:
            print('Agent updating state: ', prev_act, prev_slot)

        # разбиваем высказываение на токены
        s_t = to_tokens(user_utterance)
        slot_match = self._search_slots(s_t) # search slots
        val_match = self._search_values(s_t) # search values
        
        # examples
        # slot_match {'actor': 1.0}
        # val_match {'actor': {1: 1.0}, 'critic_rating': {}, 'genre': {}, 'mpaa_rating': {}, 'director': {}, 'release_year': {})

        for slot, values in val_match.items():
            requested = (prev_act=='request') and (prev_slot==slot)
            matched = (slot in slot_match)

            if not values:
                if requested: # asked for value but did not get it
                    if verbose:
                        print(f"Removing slot {slot} from db!")
                    self.state['database'].delete_slot(slot)
                    self.state['num_requests'][slot] = 1000
                    self.state['dont_care'].add(slot)
            else:
                for val_idx, match_ratio in values.items():
                    if verbose:
                        print ('Detected %s' % self.movie_dict.dict[slot][val_idx], ' update = ', match_ratio)

                    # expr.(5)
                    # C := upd; s_j^t[v] := match_ratio; b_j^t := slot_match; if(requested): += 1
                    if matched and requested:
                        alpha = upd * (match_ratio + 1. + slot_match[slot])
                    elif matched and not requested:
                        alpha = upd * (match_ratio + slot_match[slot])
                    elif not matched and requested:
                        alpha = upd * (match_ratio + 1.)
                    else:
                        alpha = upd * match_ratio

                    self.state['inform_slots'][slot][val_idx] += alpha
                self.state['slot_tracker'].add(slot)

    def _init_beliefs(self):
        '''инициализирует beliefs частотностями значений слотов'''
        beliefs = {s:np.copy(self.state['database'].priors[s]) 
                for s in self.state['database'].slots}
        return beliefs
