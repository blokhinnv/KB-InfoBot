import imdb
from itertools import product
import pickle
from typing import Dict
import pandas as pd


class DBCreator(object):
    def __init__(self, k: int) -> None:
        self.k = k
        self.slots = ['title', 'cast', 'rating',
                      'year', 'directors', 'aspect ratio']

        self.ia = imdb.IMDb()
        self.top_250_movies = self.ia.get_top250_movies()
        self.top_k_movies_full = [self.ia.get_movie(
            movie.movieID) for movie in self.top_250_movies[:self.k]]
        self.slot_head = 3

        self.db = []

    def fill_db(self):
        for movie in self.top_k_movies_full:
            slots_lists = []
            for slot in self.slots:
                if isinstance(movie[slot], list):
                    slots_lists.append([person['name'] for person in movie[slot]
                                        [:self.slot_head] if isinstance(person, imdb.Person.Person)])
                else:
                    slots_lists.append([movie[slot]])
            rows = list(product(*slots_lists))
            dict_rows = [dict(zip(self.slots, values)) for values in rows]
            self.db.extend(dict_rows)

    def save(self, path: str) -> None:
        with open(path, 'wb') as fp:
            pickle.dump(self.db, fp)

    def load(self, path: str) -> Dict:
        with open(path, 'rb') as fp:
            self.db = pickle.load(fp)

    def stats(self) -> Dict:
        df = pd.DataFrame(self.db)
        stats = {
            'N': df.shape[0],
            'M': df.shape[1],
            'max_j{V^j}': df.nunique().max(),
            '|M_j|': df.isna().sum().max()
        }
        return stats
