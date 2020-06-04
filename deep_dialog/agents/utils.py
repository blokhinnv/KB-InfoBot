from collections import Counter
from deep_dialog import tools
import numpy as np
import time
from typing import Dict, List

def standardize(arr):
    return arr

def calc_entropies(state: Dict, q: List, db: 'Database') -> Dict:
    # expr. (8)
    # вывод на листе 12
    entropies = {}
    for s in state.keys():
        if s not in db.slots:
            entropies[s] = 0.
        else:
            # db.ids[s] - V строк и N столбцов, (i, j) == True => в строке j стояло значение слота, имеющие в 
            # файле dicts_.json номер i
            p = (db.ids[s] * q).sum(axis=1)
            # db.unks[s]  список номеров строк, где в слоте s стояло UNK

            u = db.priors[s] * q[db.unks[s]].sum()
            c_tilde = p + u
            c_tilde = c_tilde / c_tilde.sum()
            entropies[s] = tools.entropy_p(c_tilde)
    return entropies
