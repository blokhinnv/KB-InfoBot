'''
'''

import numpy as np

class SoftDB:
    def _inform(self, probs):
        return np.argsort(probs)[::-1].tolist()

    ''' get dist over DB based on current beliefs '''
    def _check_db(self):
        '''
        Возвращает вектор длины N вероятностей того, что пользователь заинтересован в i строке
        формулы (1)-(3) см заметки лист 8
        '''
        # induce disttribution over DB based on current beliefs over slots
        probs = {}
        # матрица кол-во строк х кол-во слотов
        p_s = np.zeros((self.state['database'].N, len(self.state['database'].slots))).astype('float32')

        for i, s in enumerate(self.state['database'].slots): # по названиям слотов

            # self.state['inform_slots'] - словарь слот: список вер=тей возможных значений слота p_j[v]
            p = self.state['inform_slots'][s] / self.state['inform_slots'][s].sum()
            n = self.state['database'].inv_counts[s]  # частоты; n[-1] - частота UNK в слоте
 
            p_unk = float(n[-1]) / self.state['database'].N  # доля UNK в слоте
            p_tilde = p * (1. - p_unk) 
            p_tilde = np.concatenate([p_tilde, np.asarray([p_unk])])

            # p_tilde - вектор из M модифицированных вероятностей + (M+1) вер-ти UNK
            # self.state['database'].table[:,i] - номера значений слота i
            # p_tilde[self.state['database'].table[:,i]] - модифицированные вероятности для каждого значения
            # n[self.state['database'].table[:,i] - частоты для каждого значения
            # делим первое на второе
            p_s[:,i] = p_tilde[self.state['database'].table[:,i]]/ \
                    n[self.state['database'].table[:,i]]

        
        p_db = np.sum(np.log(p_s), axis=1)
        p_db = np.exp(p_db - np.min(p_db))
        p_db = p_db/p_db.sum()
        return p_db
