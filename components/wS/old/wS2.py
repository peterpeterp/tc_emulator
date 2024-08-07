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

class wind_estimator(object):
    def __init__(self, dir, df):
        self._dir = dir
        os.system('mkdir -p '+self._dir)
        self._tracks = df
        self._pdfs = {}


    def get_weather_distance(self, atl):
        # get pairwise distances between weather patterns
        weatherDistance = {}
        for lab in atl._grid_labels:
            tmp = np.sum((atl._axes_grid - atl._axes_grid[lab])**2, axis=1)
            weatherDistance[lab] = tmp

        # write them into a dataframe
        weather_dis = pd.DataFrame()
        weather_dis['weather_0'] = self._tracks['weather_0']
        for lab in atl._grid_labels:
            weather_dis[lab] = np.nan
            for lab2 in atl._grid_labels:
                weather_dis.loc[self._tracks.weather_0 == lab2, lab] = weatherDistance[lab][lab2]

        # calculate STD of differences for normalization
        weather_points = [atl._axes_grid[int(l)] for l in self._tracks['weather_0'].values]
        mean_weather_point = np.mean(np.array(weather_points), axis=0)
        weather_point_std = (np.sum(np.array([(p-mean_weather_point)**2 for p in weather_points])) / len(weather_points) )**0.5

        # normalize
        for lab in atl._grid_labels:
            weather_dis[lab] = weather_dis[lab] / weather_point_std # np.std(weather_dis['weather_0'])
        self._weather_dis = weather_dis

    def get_analogue_pdfs(self, atl):
        coordinates = {
            # 'sst':np.arange(26,30.2,0.2).round(2),
            'weather_0':np.array(sorted(np.unique(self._tracks['weather_0']))),
            'wind_before':np.arange(10,180,10),
            'wind_change_before':np.arange(-30,35,10),
            'wind':np.arange(10,180,10)}

        variables = [v for v in list(coordinates.keys()) if v not in ['wind']]
        pdfs = xr.DataArray( coords=coordinates, dims=variables+['wind'])

        self.get_weather_distance(atl)

        # print(sst_mod)
        modified_tracks = self._tracks.copy()

        # this is different for each sst_mod ??????
        space = xr.DataArray(modified_tracks[variables].copy().values, coords={'ID':range(self._tracks.shape[0]), 'variable':variables}, dims=['ID','variable'])
        spaceMean = space.mean('ID')
        spaceStd = space.std('ID')
        phaseSpace = (space - spaceMean) / spaceStd

        for combi in itertools.product(*[coordinates[var] for var in variables]):
            # print(combi)
            point = np.array(combi)
            point__ = (point - spaceMean) / spaceStd
            distance = (phaseSpace - point__) ** 2
            distance.loc[:,'weather_0'] = self._weather_dis[int(point[np.array(variables)=='weather_0'])].values
            distance = np.sum(distance.values, 1)

            nearest = np.argsort(distance)[:100]
            # print(space[nearest].mean('ID').values)

            winds = modified_tracks.iloc[nearest,:]['wind'].values

            kernel = stats.gaussian_kde(winds)
            pdf = kernel(pdfs.wind.values)
            pdf /= pdf.sum()
            pdfs.loc[combi] = pdf

        xr.Dataset({'pdfs':pdfs}).to_netcdf(self._dir+'/pdfs.nc')

    def load_pdfs(self):
        self._pdfs = xr.load_dataset(self._dir+'/pdfs.nc')['pdfs']

    def sample(self, conditions):
        '''
        conditions = {'sst':28.3, 'weather_0':15, 'wind_before':54, 'wind_change_before':10, 'storm_day':6}
        '''

        pdf = self._pdfs.sel(conditions, method='nearest')

        return pdf.wind.values[np.where(np.random.random() < np.cumsum(pdf.values))[0][0]]

    def save(self):
        file_name=self._dir+'/wind_obj.pkl'
        with open(file_name, 'wb') as outfile:
            pickle.dump(self, outfile)


    def resimulate_event(self, storm_id = 2017260.1231, N=100):
        '''
        atl._tracks.loc[(atl._tracks.year == 2013)].name
        atl._tracks.loc[(atl._tracks.year == 2017) & (atl._tracks.wind > 140)].sid
        storm_id = 2017277.11279
        storm_id = 2017260.1231
        storm_id = 2017242.16333
        storm_id = 2013251.13342
        '''
        obs = self._tracks.loc[self._tracks.storm == storm_id]

        plt.close('all')
        simu = np.zeros([N, obs.shape[0]])
        for i in range(N):
            simu[i, 0] = 30
            for d in range(1,obs.shape[0]):
                wind_before = simu[i,d-1]
                if d > 1:
                    wind_change_before = simu[i,d-1] - simu[i,d-2]
                else:
                    wind_change_before = 10
                conditions = {'sst':obs.sst.values[d], 'weather_0':obs.weather_0.values[d], 'wind_before':wind_before, 'wind_change_before':wind_change_before}

                new_wind = self.sample(conditions)

                simu[i,d] = new_wind
            plt.plot(simu[i,:], color='gray', linewidth=0.5, alpha=0.5)

        plt.ylim(0,160)
        plt.fill_between(range(obs.shape[0]),np.nanpercentile(simu, 13 ,axis=0), np.nanpercentile(simu, 87 ,axis=0), color='gray', alpha=0.5)
        plt.plot(np.median(simu,axis=0), color='gray', linewidth=2, alpha=1)
        plt.plot(np.mean(simu,axis=0), color='gray', linewidth=2, linestyle='--', alpha=1)
        plt.plot(obs.wind.values)
        plt.savefig(self._dir + 'storm_%s.png' % (storm_id))

        print(storm_id)
        print(obs.ACE.sum() / np.float(obs.shape[0]))
        print(np.sum(simu ** 2 / 1000, axis=1).mean() / np.float(obs.shape[0]))

    def resimulate_synth(self, synth_dict=None, N=100):
        if synth_dict is None:
            synth_dict = {
                'fav_warm' : {'sst':28.6, 'weather_0':15},
                'fav_cold' : {'sst':27.4, 'weather_0':15},
                'hamp_warm' : {'sst':28.6, 'weather_0':0},
                'hamp_cold' : {'sst':27.4, 'weather_0':0},
            }

        plt.close('all')
        for name,style in synth_dict.items():
            simu = np.zeros([N, 15])
            for i in range(N):
                simu[i, 0] = 30
                for d in range(1,15):
                    wind_before = simu[i,d-1]
                    if d > 1:
                        wind_change_before = simu[i,d-1] - simu[i,d-2]
                    else:
                        wind_change_before = 10
                    conditions = {'sst':style['sst'], 'weather_0':style['weather_0'], 'wind_before':wind_before, 'wind_change_before':wind_change_before}
                    simu[i,d] = self.sample(conditions)

            plt.plot(np.median(simu,axis=0), label=name)
        plt.legend()
        plt.savefig(self._dir + 'storm_synth.png')


    def plot_pdfs(self):

        plt.close('all')
        fig,axes = plt.subplots(nrows=3, ncols=5, figsize=(12,10))

        for r,wind_change_before in zip([0,1,2], [-15,0,15]):
            for c,wind_before in zip(range(5), [30, 50, 70, 90, 110]):
                for weather_0,lsty in zip([15,0], ['-','--']):
                    for sst,color in zip([27.2,28.6], ['c','m']):
                        conditions = {'sst':sst, 'weather_0':weather_0, 'wind_before':wind_before, 'wind_change_before':wind_change_before}
                        pdf = self._pdfs.sel(conditions, method='nearest')
                        axes[r,c].axvline(wind_before, color='k')
                        axes[r,c].plot(self._pdfs.wind, pdf, color=color, linestyle=lsty)
                        # x = pdf.wind.values[np.where(0.5 < np.cumsum(pdf.values))[0][0]]
                        # axes[r,c].axvline(x, color=color, linestyle=lsty)
        plt.tight_layout()
        plt.savefig(self._dir + 'pdfs.png')


    def plot_pdfs_sst(self):

        plt.close('all')
        fig,axes = plt.subplots(nrows=3, ncols=5, figsize=(12,10))

        for r,wind_change_before in zip([0,1,2], [-15,0,15]):
            for c,wind_before in zip(range(5), [30, 50, 70, 90, 110]):
                for sst,color in zip(np.linspace(27,29,5), sns.color_palette("plasma", 5)):
                    conditions = {'sst':sst, 'wind_before':wind_before, 'wind_change_before':wind_change_before}
                    pdf = self._pdfs.sel(conditions, method='nearest').mean('weather_0').mean('storm_day')
                    axes[r,c].axvline(wind_before, color='k')
                    axes[r,c].plot(self._pdfs.wind, pdf, color=color)
                    axes[r,c].axvline(self._pdfs.wind[np.cumsum(pdf).values >= 0.5][0], color=color)

                    # x = pdf.wind.values[np.where(0.5 < np.cumsum(pdf.values))[0][0]]
                    # axes[r,c].axvline(x, color=color, linestyle=lsty)
        plt.tight_layout()
        plt.savefig(self._dir + 'pdfs_sst.png')


'''
    wind_obj.load_pdfs()
    wind_obj.resimulate_synth()

    wind_obj.plot_pdfs()

    for storm_id in [1997246.11316,2016230.12328, 2017260.1231, 2017242.16333, 2013251.13342]:
        wind_obj.resimulate_event(storm_id)
'''



#
