'''
Created - June 3, 2016
Author - t-bhdhi
'''

class SlotReader:
    '''
    interract.py:
    slot_path = datadir + '/slot_set.txt'
    slot_set = SlotReader(slot_path)
    '''
    def __init__(self, path):
        self._load(path)
        self._invert()
        self.num_slots = len(self.slot_ids)

    def _load(self, path):
        '''
            self.slot_groups словарь название слота: номер строки
            self.slot_ids словарь название слота: является первым (0) или нет(1)
        '''
        self.slot_groups = {}
        self.slot_ids = {}
        n = 0
        f = open(path,'r')
        for line in f:
            sl = line.rstrip().split()
            i = 0
            for s in sl:
                self.slot_groups[s] = n
                self.slot_ids[s] = i # 0-head, 1-nonhead
                i = 1
            n += 1
        f.close()

    def _invert(self):
        '''self.group_slots - словарь номер строки: [список названий слотов]'''
        # create inverted index of groups to slots

        self.group_slots = {}
        for s in self.slot_ids.keys():
            if self.slot_groups[s] not in self.group_slots:
                self.group_slots[self.slot_groups[s]] = [s]
            else:
                self.group_slots[self.slot_groups[s]].append(s)
