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
        lags = 3
        weathers = np.arange(20)
        lag_weight_power={0:1,1:0.5,2:0.5}

        self._probs = xr.DataArray(0., coords={'weather_%s' %(lag):weathers for lag in range(lags)}, dims=['weather_%s' %(lag) for lag in range(lags)])
        for weather in weathers:
            tmp = self._df.loc[(self._df['weather_%s' %(0)]==weather)]
            self._probs.sel({'weather_%s' %(0):weather})[:,:] += np.sum(tmp['genesis'] > 0) / float(tmp.shape[0])

        overall_prob = np.sum(self._df['genesis'] > 0) / float(self._df.shape[0])
        for lag in range(1,lags):
            for weather in weathers:
                tmp = self._df.loc[(self._df['weather_%s' %(lag)]==weather)]
                modifier = (np.sum(tmp['genesis'] > 0) / float(tmp.shape[0])) /  overall_prob
                self._probs.sel({'weather_%s' %(lag):weather})[:,:] *= modifier ** lag_weight_power[lag]

    def sample(self, weathers, sst=None):
        return np.random.random() < self._probs.sel(weathers)

    def prob(self, weathers, sst=None):
        return self._probs.sel(weathers)

    def save(self):
        with open(self._dir+'/genesis_obj.pkl', 'wb') as outfile:
            pickle.dump(self, outfile)
