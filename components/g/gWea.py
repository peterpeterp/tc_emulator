# basics
import sys, os,pickle, inspect, textwrap, importlib, glob, itertools, inspect, resource, time
import numpy as np
import xarray as xr
import pandas as pd

import scipy
from scipy import stats


class genesis_pred(object):
    def __init__(self, dir, df):
        self._dir = dir
        os.system('mkdir -p '+self._dir)
        self._df = df

    def fit(self, atl):
        weathers = np.arange(20)
        self._probs = xr.DataArray(0., coords={'weather_0':weathers}, dims=['weather_0'])
        for weather in weathers:
            tmp = self._df.loc[(self._df['weather_%s' %(0)]==weather)]
            self._probs.loc[weather] = np.sum(tmp['genesis'] > 0) / float(tmp.shape[0])

    def sample(self, weather, sst=None):
        return np.random.random() < self._probs.loc[weather].values

    def prob(self, weather, sst=None):
        return self._probs.loc[weather]

    def save(self):
        with open(self._dir+'/genesis_obj.pkl', 'wb') as outfile:
            pickle.dump(self, outfile)
