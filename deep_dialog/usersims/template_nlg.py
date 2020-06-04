'''

Takes user action and templates from file and produces NL utterance. 
'''

import pickle as pkl
import json
import random
from typing import Dict

class TemplateNLG:
    def __init__(self, template_file):
        self.templates = json.load(open(template_file, 'r'))

    def generate(self, act: str, request_slots: Dict, inform_slots: Dict):
        n_r = len(request_slots.keys())
        i_slots = {k: v for k,v in inform_slots.items() if v is not None}
        n_i = len(i_slots.keys())
        key = '%s_%d_%d' % (act, n_r, n_i)

        temp = random.choice(self.templates[key])
        sent = self._fill_slots(temp, request_slots, i_slots)

        return sent

    def _fill_slots(self, temp, request_slots, i_slots):
        reqs = list(request_slots.keys())
        infs = list(i_slots.keys())
        random.shuffle(reqs)
        random.shuffle(infs)

        for i,k in enumerate(reqs):
            temp = temp.replace('@rslot%d'%i, k)

        for i,k in enumerate(infs):
            temp = temp.replace('@islot%d'%i, k)
            temp = temp.replace('@ival%d'%i, i_slots[k])

        return temp
