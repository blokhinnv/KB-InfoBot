'''

a rule-based user simulator

'''

import argparse, json, time
import random
import copy
import nltk
from typing import Tuple, Dict

from deep_dialog import dialog_config

DOMAIN_NAME = 'movie'

def weighted_choice(choices, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(choices,weights):
        if upto + w >= r:
            return c
        upto += w
    assert False, "shouldnt get here"

class RuleSimulator:
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, 
            start_set=None, max_turn=20, nlg=None, err_prob=0., db=None, 
            dk_prob=0., sub_prob=0., max_first_turn=5, verbose=False):
        self.max_turn = dialog_config.MAX_TURN
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        self.nlg = nlg
        self.err_prob = err_prob
        self.database = db
        self.dk_prob = dk_prob
        self.sub_prob = sub_prob
        self.max_first_turn = max_first_turn

        self.goal = {}
        self.state = {}
        self.verbose = verbose

    ''' randomly sample a start state '''
    def _sample_action(self) -> Tuple[bool, Dict]:
        self.state = {}
        '''
            Берет текущее действие случайным образом из dialog_config.start_dia_acts (а там только request)
            если хотя бы один из словарей goal['inform_slots'] или goal['request_slots'] заполнен:
                Если заполнен inform_slots, то берет несколько не None названий слотов 
                и копирует из goal[inform_slots][s] в state[inform_slots][s]

                Если заполнен request_slots, то берется случайный ключ оттуда 
                и в state['request_slots'][request_slot] пишет UNK

            Если диалог не завершается !self.state['diaact'] in ['thanks','closing'], 
                то corrupt: добавляем словарь self.state['inform_slots_noisy']

            Генерирует ответ на языке, возвращает флаг (диалог закончен?) и словарь self.state
        '''
        # self.state['diaact'] строка \in {inform, request, UNK, deny, thanks,closing, ..}
        self.state['diaact'] = random.choice(list(dialog_config.start_dia_acts.keys()))
        self.state['turn'] = 0
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['prev_diaact'] = 'UNK'
        
        # если хотя бы один из goal['inform_slots'] или goal['request_slots'] заполнен:
        if (len(self.goal['inform_slots']) + len(self.goal['request_slots'])) > 0:
            if len(self.goal['inform_slots']) > 0:
                care_about = [s for s,v in self.goal['inform_slots'].items() if v is not None]
                if self.verbose:
                    print('У искомой сущности есть слоты (в неполной БД)', care_about)
                known_slots = random.sample(care_about, 
                        random.randint(1,min(self.max_first_turn,len(care_about))))
                for s in known_slots:
                    self.state['inform_slots'][s] = self.goal['inform_slots'][s]
                if self.verbose:
                    print('Пользователь уже назвал (?):', self.state['inform_slots'])    
            
            if len(self.goal['request_slots']) > 0:
                request_slot = random.choice(list(self.goal['request_slots'].keys()))
                self.state['request_slots'][request_slot] = 'UNK'
        
        if (self.state['diaact'] in ['thanks','closing']): 
            episode_over = True
        else: 
            episode_over = False

        if not episode_over:
            self.corrupt()
        
        sent = self.nlg.generate(self.state['diaact'], 
                                 self.state['request_slots'],
                                 self.state['inform_slots_noisy']) if self.nlg is not None else ''
        self.state['nl_sentence'] = sent
        self.state['episode_over'] = episode_over
        self.state['reward'] = 0
    
        return episode_over, self.state

    ''' sample a goal '''
    def _sample_goal(self) -> None:

        if self.start_set is not None:
            self.goal = random.choice(self.start_set)  # sample user's goal from the dataset
        else:
            '''
                Берем случайный номер от 0 до N-1 -- номер искомой сущности
                Находим слоты, которые у нее заполнены (known_slots), 
                из них берем несколько пропорционально self.dk_prob (care_about)
                Идем по всем слотам s, если он в care_about, 
                то добавляет его значение в self.goal['inform_slots'][s], иначе добавляем туда None
                Если все оказались None, то добавляем в self.goal['inform_slots'] первый случайный заполненный слот
                Получаем словарь self.goal
            '''
            # sample a DB record as target
            self.goal = {}
            self.goal['request_slots'] = {}
            self.goal['request_slots'][DOMAIN_NAME] = 'UNK'

            self.goal['target'] = random.randint(0,self.database.N-1)
            self.goal['inform_slots'] = {}

            known_slots = [s for i,s in enumerate(dialog_config.inform_slots) 
                    if self.database.tuples[self.goal['target']][i]!='UNK']

            care_about = random.sample(known_slots, int(self.dk_prob*len(known_slots)))

            for i,s in enumerate(self.database.slots):
                if s not in dialog_config.inform_slots: 
                    continue
                val = self.database.tuples[self.goal['target']][i]
                if s in care_about and val != 'UNK':
                    self.goal['inform_slots'][s] = val
                else:
                    self.goal['inform_slots'][s] = None

            if all([v==None for v in self.goal['inform_slots'].values()]):
                while True:
                    s = random.choice(self.goal['inform_slots'].keys())
                    i = self.database.slots.index(s)
                    val = self.database.tuples[self.goal['target']][i]
                    if val != 'UNK':
                        self.goal['inform_slots'][s] = val
                        break

    def print_goal(self):
        print ('User target = ', ', '.join(['%s:%s' %(s,v) for s,v in \
                zip(['movie']+self.database.slots, \
                [self.database.labels[self.goal['target']]] + \
                self.database.tuples[self.goal['target']])]))
        print ('User information = ', ', '.join(['%s:%s' %(s,v) for s,v in \
                self.goal['inform_slots'].items() if v is not None]), '\n')

    ''' initialization '''
    def initialize_episode(self) -> Dict:
        self._sample_goal()
        
        # first action
        episode_over, user_action = self._sample_action()
        assert (episode_over != 1),' but we just started'
        return user_action

    ''' update state: state is sys_action '''
    def next(self, state: Dict) -> Tuple[Dict, bool, float]:
        '''
            state = agent.act_state from next()
            Если диалог слишком долгий - конец (reward=FAILED_DIALOG_REWARD)
                Если от агента пришла команда Inform, то в полученных результатах поиска ищется номер сущности, которую
                загадывал пользоваль. Если она по рангу ниже SUCCESS_MAX_RANK, то она уменьшается пропорционально рангу,
                а если выше - то reward=FAILED_DIALOG_REWARD

                Если от агента пришла команда Request, то пользователь "называет" запрашиваемый слот (из goal[inform_slots]),
                награда r =PER_TURN_REWARD

            Если диалог не завершается, 
                то corrupt: добавляем словарь self.state['inform_slots_noisy']

            Генерирует ответ на языке, возвращает флаг (диалог закончен?) и словарь self.state            
        '''
        self.state['turn'] += 1
        reward = 0
        episode_over = False
        self.state['prev_diaact'] = self.state['diaact']
        self.state['inform_slots'].clear()
        self.state['request_slots'].clear()
        self.state['inform_slots_noisy'].clear()
        
        if (self.max_turn > 0 and self.state['turn'] >= self.max_turn):
            reward = dialog_config.FAILED_DIALOG_REWARD
            episode_over = True
            self.state['diaact'] = 'deny'
        else:
            act = state['diaact']
            if act == 'inform':
                episode_over = True
                goal_rank = state['target'].index(self.goal['target'])
                if self.verbose:
                    print(f"Просил {self.goal['target']}, нашли на позиции {goal_rank}")
                if goal_rank < dialog_config.SUCCESS_MAX_RANK:
                    reward = dialog_config.SUCCESS_DIALOG_REWARD * \
                            (1.-float(goal_rank)/dialog_config.SUCCESS_MAX_RANK)
                    self.state['diaact'] = 'thanks'
                else:
                    reward = dialog_config.FAILED_DIALOG_REWARD
                    self.state['diaact'] = 'deny'
            elif act == 'request':
                slot = list(state['request_slots'].keys())[0]
                if slot in self.goal['inform_slots']:
                    self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                else:
                    self.state['inform_slots'][slot] = None
                self.state['diaact'] = 'inform'
                reward = dialog_config.PER_TURN_REWARD

        if not episode_over:
            self.corrupt()
        
        sent = self.nlg.generate(self.state['diaact'],self.state['request_slots'],
                self.state['inform_slots_noisy']) if self.nlg is not None else ''
        # print('sent=', sent)
        self.state['nl_sentence'] = sent
        self.state['episode_over'] = episode_over
        self.state['reward'] = reward

        return self.state, episode_over, reward

    ''' user may make mistakes '''
    def corrupt(self) -> None:
        '''
            для каждого слота slot self.goal['inform_slots']
                если его значение None, то так и пишем в self.state['inform_slots_noisy'][slot]
                подбрасываем монетку; 
                    если да, то в self.state['inform_slots_noisy'][slot] пишем 
                    случайное возможное значение этого слота. 
                    если нет -- то self.state['inform_slots_noisy'][slot] 
                
                подбрасываем монетку;
                    если да, чуть-чуть портим значение в self.state['inform_slots_noisy'][slot]
        '''

        self.state['inform_slots_noisy'] = {}
        for slot in self.state['inform_slots'].keys():
            if self.state['inform_slots'][slot] is None:
                self.state['inform_slots_noisy'][slot] = None
            else:
                prob_sub = random.random()
                if prob_sub < self.sub_prob: # substitute value
                    self.state['inform_slots_noisy'][slot] = \
                            random.choice(self.movie_dict.dict[slot])
                else:
                    self.state['inform_slots_noisy'][slot] = self.state['inform_slots'][slot]
                prob_err = random.random()
                if prob_err < self.err_prob: # corrupt value
                    self.state['inform_slots_noisy'][slot] = \
                            self._corrupt_value(self.state['inform_slots_noisy'][slot])

    def _corrupt_value(self, val: str) -> str:
        '''
            Если в val больше 1 токена, удаляем 1 случайный. 
            Оставшиеся немножко меняем, если это числа
            Склеиваем пробелами и возвращает
        '''
        def _is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        def _is_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        tokens = nltk.word_tokenize(val)
        if len(tokens)>1: 
            tokens.pop(random.randrange(len(tokens)))
        out = []
        for t in tokens:
            if _is_int(t):
                out.append(str(int(random.gauss(int(t),0.5))))
            elif _is_float(t):
                out.append('%.1f' %random.gauss(float(t),0.5))
            else:
                out.append(t)
        return ' '.join([o for o in out])

    ''' user state representation '''
    def stateVector(self, action):
        vec = [0]*(len(self.act_set.dict) + len(self.slot_set.slot_ids)*2)

        if action['diaact'] in self.act_set.dict.keys(): vec[self.act_set.dict[action['diaact']]] = 1
        for slot in action['slots'].keys():
            slot_id = self.slot_set.slot_ids[slot] * 2 + len(self.act_set.dict)
            slot_id += 1
            if action['slots'][slot] == 'UNK': vec[slot_id] =1

        return vec

    ''' print the state '''
    def print_state(self, action):
        stateStr = 'Turn %d user action: %s, history slots: %s, inform_slots: %s, request slots: %s, rest_slots: %s' % (action['turn'], action['diaact'], action['history_slots'], action['inform_slots'], action['request_slots'], action['rest_slots'])
        print( stateStr)



def main(params):
    user_sim = RuleSimulator()
    user_sim.initialize_episode()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print ('User Simulator Parameters: ')
    print (json.dumps(params, indent=2))

    main(params)
