# basics
import sys, os,pickle, inspect, textwrap, importlib, glob, itertools, inspect, resource, time
import numpy as np
import xarray as xr
import pandas as pd

import scipy
from scipy import stats

exec("import %s; importlib.reload(%s); from %s import *" % tuple(['components.windSpeed._helping_functions']*3))

'''
nearest neighbors with
weather lag 2
SST
'''

class genesis_pred(object):
    def __init__(self, dir, df):
        self._dir = dir
        os.system('mkdir -p '+self._dir)
        self._tracks = df
        self._pdfs = {}

    def fit(self, atl):
        coordinates = {
            'sst':np.arange(26,30.2,0.2).round(2),
            'weather_0':np.arange(0,20,1),
            'weather_1':np.arange(0,20,1),
            'weather_2':np.arange(0,20,1)
        }

        variables = [v for v in list(coordinates.keys())]
        self._probs = xr.DataArray( coords=coordinates, dims=variables)

        weather_dis = get_weather_distance(self._tracks, atl)

        modified_tracks = self._tracks.copy()

        space = xr.DataArray(modified_tracks[variables].copy().values, coords={'ID':range(self._tracks.shape[0]), 'variable':variables}, dims=['ID','variable'])
        spaceMean = space.mean('ID')
        spaceStd = space.std('ID')
        phaseSpace = (space - spaceMean) / spaceStd

        for combi in itertools.product(*[coordinates[var] for var in variables]):
            # print(combi)
            point = np.array(combi)
            point__ = (point - spaceMean) / spaceStd
            distance = (phaseSpace - point__) ** 2
            distance.loc[:,'weather_0'] = weather_dis[int(point[np.array(variables)=='weather_0'])].values
            distance.loc[:,'weather_1'] = weather_dis[int(point[np.array(variables)=='weather_1'])].values
            distance.loc[:,'weather_2'] = weather_dis[int(point[np.array(variables)=='weather_2'])].values
            distance = np.sum(distance.values, 1)

            nearest = np.argsort(distance)[:100]
            self._probs.loc[combi] = modified_tracks.iloc[nearest,:]['genesis'].values.mean()

    def sample(self, conditions, sst=None):
        return np.random.random() < self._probs.sel(conditions, method='nearest')

    def prob(self, weathers, sst=None):
        return self._probs.sel(weathers)

    def save(self):
        with open(self._dir+'/genesis_obj.pkl', 'wb') as outfile:
            pickle.dump(self, outfile)
