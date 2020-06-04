import pickle
import json
import copy
import nltk
import string

from collections import defaultdict
from deep_dialog.tools import to_tokens

# import cPickle as pickle
# dict_data = pickle.load(open('D:/nvb/GODS_NG/KB-InfoBot/data/imdb-XL/dicts.json', 'rb'))
# import json
# json.dump(dict_data, open('D:/nvb/GODS_NG/KB-InfoBot/data/imdb-XL/dicts_.json', 'w', encoding='utf8'), ensure_ascii=False, indent=2)


class MovieDict:
    '''
        Использование в interact
        datadir = './data/' + params['dataset']
        dict_path = datadir + '/dicts.json'
        movie_kb = MovieDict(dict_path)
    '''
    def __init__(self, path):
        self.load_dict(path)
        self.count_values()
        self._build_token_index()
    
    def load_dict(self, path):
        dict_data = json.load(open(path, 'r'))
        self.dict = copy.deepcopy(dict_data)


    def count_values(self) -> None:
        '''
            по этому куску видно, что self.dict имеет вид ~ slot: vals:=Iterable[Any]
            считает количество значений для слота
        '''
        self.lengths = {}
        for k, v in self.dict.items():
            self.lengths[k] = len(v)

    def _build_token_index(self):
        '''
            строит словарь слот: {
                токен: номер значения данного слота, в котором встречается этот токен 
            }
        '''
        self.tokens = {}
        for slot, vals in self.dict.items():
            self.tokens[slot] = defaultdict(list)
            for value_idx, value in enumerate(vals):
                # разбиваем каждое значение на токены без пунктуации и стопслов
                tokens_from_value = to_tokens(value)
                for token in tokens_from_value: 
                    self.tokens[slot][token].append(value_idx)
