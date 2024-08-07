# basics
import sys, os,pickle, inspect, textwrap, importlib, glob, itertools, inspect, resource, time
import numpy as np
import xarray as xr
import pandas as pd

import scipy
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import seaborn as sns
import matplotlib.pyplot as plt

# from components.windSpeed._helping_functions import *
exec("import %s; importlib.reload(%s); from %s import *" % tuple(['components.wS._helping_functions']*3))

class wind_estimator(object):
    def __init__(self, dir, df):
        self._dir = dir
        os.system('mkdir -p '+self._dir)
        self._tracks = df
        self._pdfs = {}

    def get_analogue_pdfs(self, atl):
        coordinates = {
            'sst':np.arange(26,30.2,0.2).round(2),
            'weather_0':np.array(sorted(np.unique(self._tracks['weather_0']))),
            'wind_before':np.arange(10,180,10),
            # 'wind_change_before':np.arange(-30,35,10),
            'wind':np.arange(10,180,10)}
        NN=50

        variables = [v for v in list(coordinates.keys()) if v not in ['wind']]
        pdfs = xr.DataArray( coords=coordinates, dims=variables+['wind'])

        # get weather distances
        weather_dis = get_weather_distance(self._tracks, atl)

        # prepare array for saving distances of NN
        coords = {k:v for k,v in coordinates.items() if k in variables}
        coords['d'] = variables
        coords['q'] = [0,10,17,50,83,90,100]
        NNdistances = xr.DataArray( coords=coords, dims=variables+['d','q'])

        # get space from which distances will be calculated
        space = xr.DataArray(self._tracks[variables].copy().values, coords={'ID':range(self._tracks.shape[0]), 'variable':variables}, dims=['ID','variable'])
        spaceMean = space.mean('ID')
        spaceStd = space.std('ID')
        # standardize
        phaseSpace = (space - spaceMean) / spaceStd

        for combi in itertools.product(*[coordinates[var] for var in variables]):
            # print(combi)
            point = np.array(combi)
            point__ = (point - spaceMean) / spaceStd
            # get distanes
            distance = (phaseSpace - point__) ** 2
            distance.loc[:,'weather_0'] = weather_dis[int(point[np.array(variables)=='weather_0'])].values
            distance = np.sum(distance.values, 1)

            # identify 100 NN
            nearest = np.argsort(distance)[:NN]
            # store distances
            combi__ = list(combi)
            tmp__ = np.nanpercentile(space[nearest],NNdistances.q.values, axis=0) - combi__
            NNdistances.loc[combi] = tmp__.T
            # get winds of 100 NN
            winds = self._tracks.iloc[nearest,:]['wind'].values
            # estimate pdf for 100 NN
            kernel = stats.gaussian_kde(winds)
            pdf = kernel(pdfs.wind.values)
            pdf /= pdf.sum()
            pdfs.loc[combi] = pdf


        xr.Dataset({'pdfs':pdfs}).to_netcdf(self._dir+'/pdfs.nc')
        xr.Dataset({'distances':NNdistances}).to_netcdf(self._dir+'/distances.nc')

        # BC
        self._BC = xr.DataArray(0., coords={'wind_before':coordinates['wind_before']}, dims=['wind_before'])
        tmp = NNdistances.sel({'d':'wind_before','q':50})
        for k in [k for k in tmp.dims if k not in ['wind_before']]:
            tmp = tmp.mean(k)
        self._BC.values = - tmp.values
        print(self._BC)

    def load_pdfs(self):
        self._pdfs = xr.load_dataset(self._dir+'/pdfs.nc')['pdfs']



    def sample(self, conditions):
        '''
        conditions = {'sst':28.3, 'weather_0':15, 'wind_before':54, 'wind_change_before':10, 'storm_day':6}
        '''

        pdf = self._pdfs.sel(conditions, method='nearest')
        BC = self._BC.sel({'wind_before':conditions['wind_before']}, method='nearest')
        return pdf.wind.values[np.where(np.random.random() < np.cumsum(pdf.values))[0][0]] + BC

    def save(self):
        file_name=self._dir+'/wind_obj.pkl'
        with open(file_name, 'wb') as outfile:
            pickle.dump(self, outfile)


