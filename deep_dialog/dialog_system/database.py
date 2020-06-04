'''

Class for database
Пример обращения: (interact.py)
    datadir = './data/' + params['dataset']
    db_full_path = datadir + '/db.txt'
    movie_kb = MovieDict(dict_path)
    db_full = Database(db_full_path, movie_kb, name=params['dataset'])
'''

import csv
import io
import numpy as np
import nltk
import time
from typing import Tuple, List

from collections import defaultdict
from deep_dialog import dialog_config
from deep_dialog.tools import to_tokens

class Database:
    def __init__(self, path: str, dicts: 'MovieDict', name=''):
        '''
            path - путь к tsv базе (db.txt)
            dicts: MovieDict 

        '''
        self.path = path
        self.name = name
        self._load_db(path)
        self._shuffle()
        self._build_inv_index(dicts)
        self._build_table(dicts)
        self._get_priors()
        self._prepare_for_entropy(dicts)
        self._prepare_for_search()

    def _load_db(self, path):
        '''
            Считывает tsv БД и создает 2 списка: self.labels (названия фильмов) & self.tuples (остальные хар-ки)
        '''
        try:
            fi = io.open(path,'r')
            # первая строка - названия слотов
            self.slots = fi.readline().rstrip().split('\t')[1:]
            # список списков со значениями слотов
            tupl = [line.rstrip().split('\t') for line in fi]
            self.labels = [t[0] for t in tupl]  # названия фильмов
            self.tuples = [t[1:] for t in tupl] # остальные хар-ки
            fi.close()
        except UnicodeDecodeError:
            # все то же, но без io
            fi = open(path, 'r')
            self.slots = fi.readline().rstrip().split('\t')[1:]
            tupl = [line.rstrip().split('\t') for line in fi]
            self.labels = [t[0] for t in tupl]
            self.tuples = [t[1:] for t in tupl]
            fi.close()
        self.N = len(self.tuples)

    def _shuffle(self):
        '''
         для self.slots и self.tuples меняет порядок слотов так, чтобы они соответствовали конфигу
        ''' 
        #  как слоты перечислены в конфиге
        index = [self.slots.index(s) for s in dialog_config.inform_slots]
        self.slots = [self.slots[ii] for ii in index]
        self.tuples = [[row[ii] for ii in index] for row in self.tuples]

    def lookup(self, query: List, match_unk=True) -> Tuple[List, List]:
        '''нужно смотреть deep_dialog/agents/hardDB'''
        '''query: список значений для слотов'''
        def eq_w_UNK(t1, t2):
            for i in range(len(t1)):
                if t1[i] != t2[i] and t1[i] != 'UNK' and t2[i] != 'UNK':
                    return False
            return True

        # индексы заполненных в query слотов
        col_idxs = [value_idx for value_idx, value in enumerate(query) if value is not None]
        # берем вертикальный срез таблицы по заполненным слотам
        c_db = [[row[col_idx] for col_idx in col_idxs] for row in self.tuples]
        # значения для поиска
        c_q = [query[col_idx] for col_idx in col_idxs]
        # индексы строк, которые совпадают с искомыми
        if match_unk: 
            row_match_idx = [row_idx for row_idx, row in enumerate(c_db) if eq_w_UNK(row, c_q)]
        else: 
            row_match_idx = [row_idx for row_idx, row in enumerate(c_db) if row==c_q]

        # сами строки
        results = [self.tuples[ii] for ii in row_match_idx]
        return results, row_match_idx

    def delete_slot(self, slot):
        try:
            slot_index = self.slots.index(slot)
        except ValueError:
            print ('Slot not found!!!')
            return

        for row in self.tuples: 
            del row[slot_index]
        self.table = np.delete(self.table, slot_index, axis=1)
        self.counts = np.delete(self.counts, slot_index, axis=1)
        del self.slots[slot_index]

    def _build_inv_index(self, dicts: 'MovieDict'):
        '''
            Для каждого слота:
                для каждого значения находит его номер в dicts_.json,
                дописывает этот номер в self.inv_index[slot][v] 
                [для слота slot значение v указано в строках БД c номерами self.inv_index[slot][v]]
                и
                суммирует количество раз, которое этот номер находился в self.inv_counts[slot][v_id] 
                (частота встречаемости значения в данном слоте, частота встречаемости значения UNK - последний эл-т)
                [для слота slot значение, которое стоит на позиции v_idx в dicts_.json[slot], 
                 встретилось db_full.inv_counts[slot][v_idx] раз] 
               
            Создаются
            self.inv_index, self.inv_count
        '''
        self.inv_index = {}
        self.inv_counts = {}
        for i, slot in enumerate(self.slots):
            V = dicts.lengths[slot] # количество возможных значений слота slot
            self.inv_index[slot] = defaultdict(list)
            self.inv_counts[slot] = np.zeros((V+1, )).astype('float32')

            # все значения слота slot
            values = [t[i] for t in self.tuples]
            for j, v in enumerate(values):
                v_id = dicts.dict[slot].index(v) if v != 'UNK' else V
                self.inv_index[slot][v].append(j)
                self.inv_counts[slot][v_id] += 1

    def _build_table(self, dicts):
        '''
            Создает 2 таблицы размером кол-во строк в БД х кол-во слотов
            для строки i и слота j находит номер значения слота в dicts_.json и
            в self.table на месте i,j стоит номер значения v:=DB(i, j) в данном слоте j
            (если знчение неизвестно - то кол-во значений интента)
            в self.counts на месте i,j стоит частота встречаемости значения v:=DB(i, j) в данном слоте j
        '''
        self.table = np.zeros((len(self.tuples),len(self.slots))).astype('int16')
        self.counts = np.zeros((len(self.tuples),len(self.slots))).astype('float32')
        for i, t in enumerate(self.tuples):
            for j, v in enumerate(t):
                s = self.slots[j]
                self.table[i,j] = dicts.dict[s].index(v) if v!='UNK' else dicts.lengths[s]
                self.counts[i,j] = self.inv_counts[s][self.table[i,j]]

    def _get_priors(self):
        '''
            Нормирует частоты self.inv_count на сумму частот (кроме частоты UNK)
        '''
        self.priors = {slot: (self.inv_counts[slot][:-1] / self.inv_counts[slot][:-1].sum())
                       for slot in self.slots}

    def _prepare_for_entropy(self, dicts: 'MovieDict'):
        self.ids = {}
        self.ns = {}
        self.non0 = {}
        self.unks = {}

        for i, s in enumerate(self.slots):
            
            V = dicts.lengths[s]  # количество значений слота s
            db_c = self.table[:, i]  # номера значений слота s 
            self.unks[s] = np.where(db_c == V)[0]  # список номеров строк, где в слоте s стояло UNK
            # self.priors[s] имеет размерность = кол-во различных значений в слоте 
            # self.N - кол-во строк

            # self.ids[s] - V строк и N столбцов, (i, j) == True => в строке j стояло значение слота, имеющие в 
            # файле dicts_.json номер i
            self.ids[s] = (np.mgrid[:self.priors[s].shape[0],:self.N]==db_c)[0]
            # self.ns[s][i] - кол-во раз, которое значение с номером i встретилось в слоте s
            self.ns[s] = self.ids[s].sum(axis=1)
            # self.non0[s] - номера значений, которые встречались в слоте s 
            self.non0[s] = np.nonzero(self.ns[s])[0]

    def _prepare_for_search(self):
        '''Разбивает названия слотов на токены'''
        self.slot_tokens = {}
        for slot in self.slots:
            self.slot_tokens[slot] = to_tokens(slot)
