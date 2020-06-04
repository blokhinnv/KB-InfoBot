'''
'''
from typing import Tuple, List, Dict

from deep_dialog import dialog_config, tools
from deep_dialog.agents.agent import Agent

import operator
import random
import numpy as np

class AgentActRule(Agent):

    def initialize_episode(self) -> None:
        '''Начало диалога после создания менеджера'''
        self.state = {}
        self.state['diaact'] = 'UNK'
        self.state['inform_slots'] = {}
        self.state['turn'] = 0

    ''' update agent state '''
    def _update_state(self, user_action):
        '''Обновляет состояние агента'''
        for s in user_action['inform_slots'].keys():
            self.state['inform_slots'][s] = user_action['inform_slots'][s]
    
    ''' get next action based on rules '''
    def next(self, user_action, verbose=False) -> Dict:
        '''
            Обновляется состояние агента информацией от пользователя
            Производится жесткий поиск по БД
            Если 
                - ничего не нашлось => запросить другой слот
                - если в запросе участвовали все слоты либо нашелся 1 вариант => вернуть ответ
                - в запросе участвовали не все слоты и нашелся не один ответ => 
                    для каждого неизвестного агенту слота считает энтропию
                    если макс. энтропия положительная - запросить слот с макс. энтропией, иначе вернуть ответы
                для найденных жестким поиском сущностей вероятность act[posterior] равномерно размазывается
        '''
        self._update_state(user_action)

        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        # db_status - список найденных при поиске списков-строк со значениями, db_index - список номеров строк
        db_status, db_index = self._check_db()
        print(f'Нашло {len(db_status)} строк')
        if not db_status:
            # no match, some error, re-ask some slot
            # ничего не нашлось => запросить другой слот
            act['diaact'] = 'request'
            request_slot = random.choice(self.state['inform_slots'].keys())
            act['request_slots'][request_slot] = 'UNK'
            print('ничего не нашлось => запросить другой слот')

        elif (len(self.state['inform_slots']) == len(dialog_config.sys_request_slots)) or len(db_status)==1:
            # если в запросе участвовали все слоты либо нашелся 1 вариант => вернуть ответ
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_index)
            print('если в запросе участвовали все слоты либо нашелся 1 вариант => вернуть ответ')

        else:
            # в запросе участвовали не все слоты и нашелся не один ответ
            # request a slot not known with max entropy

            # слоты, которые знает пользователь
            known_slots = self.state['inform_slots'].keys()
            unknown_slots = [s for s in dialog_config.sys_request_slots if s not in known_slots]
            slot_entropy = {}

            # для каждого неизвестного пользователю слота считает энтропию ПО НАЙДЕННЫМ СТРОКАМ!
            for s in unknown_slots:
                db_idx = self.database.slots.index(s)
                db_matches = [m[db_idx] for m in db_status]
                slot_entropy[s] = tools.entropy(db_matches)

            # слот с максимальной энтропией, максимальная энтропия
            request_slot, max_ent = max(slot_entropy.items(), key=operator.itemgetter(1))
            # если макс. энтропия положительная - спросить слот с макс. энтропией
            # иначе вернуть ответы
            print('в запросе участвовали не все слоты и нашелся не один ответ')
            print('Энтропии: ', slot_entropy)
            if max_ent > 0.:
                act['diaact'] = 'request'
                act['request_slots'][request_slot] = 'UNK'
            else:
                act['diaact'] = 'inform'
                act['target'] = self._inform(db_index)

        # len(self.database.labels) = кол-во строк
        act['posterior'] = np.zeros((len(self.database.labels),))
        # для найденных жестким поиском сущностей вероятность равномерно размазывается
        act['posterior'][db_index] = 1./len(db_index)

        return act

    def terminate_episode(self, user_action):
        return

    def _inform(self, db_index: List) -> List:
        '''Перемешивает найденные индексы строк и дописывает после них все остальные номера строк'''
        target = db_index
        if len(target) > 1: 
            random.shuffle(target)
        full_range = list(range(self.database.N))
        random.shuffle(full_range)
        for i in full_range:
            if i not in db_index: 
                target.append(i)
        return target

    ''' query DB based on current known slots '''
    def _check_db(self) -> Tuple[List, List]:
        # from query to db form current inform_slots
        db_query = []
        for s in self.database.slots:
            if s in self.state['inform_slots']:
                db_query.append(self.state['inform_slots'][s])
            else:
                db_query.append(None)
        print('query for lookup', db_query)
        matches, indexes = self.database.lookup(db_query)
        return matches, indexes

    ''' sample value from current state of database '''
    def _sample_slot(self, slot, matches):
        if not matches:
            return None
        index = self.database.slots.index(slot)
        return random.choice([m[index] for m in matches])
