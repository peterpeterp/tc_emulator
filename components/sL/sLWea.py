# basics
import sys, os,pickle, inspect, textwrap, importlib, glob, itertools, inspect, resource, time
import numpy as np
import xarray as xr
import pandas as pd

import scipy
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

class storm_length_estimator(object):
    def __init__(self, dir, atl, tracks):
        self._dir = dir
        os.system('mkdir -p '+self._dir)


        storms = tracks.loc[(tracks.genesis==1), 'storm']
        storm_length = np.array([tracks.loc[(tracks.storm == sto)].shape[0] for sto in storms])
        x = np.arange(1,np.max(storm_length)+1)

        self._pdfs = xr.DataArray(coords={'weather_0':np.unique(atl._clust_labels), 'stormL':x}, dims=['weather_0','stormL'])

        plt.close('all')
        fig,axes = plt.subplots(nrows=atl._nrows, ncols=atl._ncols, figsize=(atl._nrows*2,atl._ncols*2), sharex=True, sharey=True)
        for lab in np.unique(atl._clust_labels):
            r,c = tuple(atl._axes_grid[atl._grid_labels==lab,:][0])
            ax = axes[r,c]
            ax.annotate(lab, xy=(0,1), xycoords='axes fraction', va='top')
            if r==atl._nrows:
                ax.set_xlabel('storm length [days]')

            storms_in_weather = tracks.loc[(tracks.genesis==1) & (tracks.weather_0==lab), 'storm']
            storm_length = np.array([tracks.loc[(tracks.storm == sto)].shape[0] for sto in storms_in_weather])

            kernel = stats.gaussian_kde(storm_length)
            pdf_fitted = kernel(x)
            pdf_fitted /= pdf_fitted.sum()

            self._pdfs[lab] = pdf_fitted
            ax.plot(x,np.cumsum(pdf_fitted), c='r')
            ax.plot([1,np.median(storm_length),np.median(storm_length)],[0.5,0.5,0], c='r')
            ax.annotate(np.median(storm_length), xy=(np.median(storm_length),0.2), c='r', ha='left')
            ax.plot([1,np.percentile(storm_length,80),np.percentile(storm_length,80)],[0.8,0.8,0], c='r')
            ax.annotate(np.percentile(storm_length,80), xy=(np.percentile(storm_length,80),0.7), c='r', ha='left')

            ax.set_xlim(0,30)
            ax.set_ylim(0,1)
        plt.savefig(self._dir+'/end_kernel.png', bbox_inches='tight'); plt.close()

    def sample(self,conds):
        return np.where(np.random.random() < np.cumsum(self._pdfs.sel(conds)))[0][0] + 1

    def save(self):
        with open(self._dir+'/end_obj.pkl', 'wb') as outfile:
            pickle.dump(self, outfile)

    def plot_simulated_storm_length(self, atl, tracks):
        plt.close('all')
        fig,axes = plt.subplots(nrows=atl._nrows, ncols=atl._ncols, figsize=(atl._nrows*2,atl._ncols*2), sharex=True, sharey=True)
        for lab in np.unique(atl._clust_labels):
            r,c = tuple(atl._axes_grid[atl._grid_labels==lab,:][0])
            ax = axes[r,c]
            ax.annotate(lab, xy=(0,1), xycoords='axes fraction', va='top')
            if r==atl._nrows:
                ax.set_xlabel('storm length [days]')

            labels = np.where(np.isin(atl._axes_grid[:,0],range(r-1,r+2)) & np.isin(atl._axes_grid[:,1],range(c-1,c+2)))[0]
            storms_from_region = tracks.loc[(tracks.genesis==1) & np.isin(tracks.weather_0,labels), 'storm']
            storm_length = np.array([tracks.loc[(tracks.storm == sto)].shape[0] for sto in storms_from_region])
            ax.hist(storm_length, bins=np.arange(1,30), density=True, alpha=0.5)

            storm_length = np.array([self.sample({'weather_0':lab}) for _ in range(10000)])
            ax.hist(storm_length, bins=np.arange(1,30), density=True, alpha=0.5)
        plt.savefig(self._dir+'/sampled_storm_length.png', bbox_inches='tight'); plt.close()

        plt.close('all')
        fig,axes = plt.subplots(nrows=atl._nrows, ncols=atl._ncols, figsize=(atl._nrows*2,atl._ncols*2), sharex=True, sharey=True)
        for lab in np.unique(atl._clust_labels):
            r,c = tuple(atl._axes_grid[atl._grid_labels==lab,:][0])
            ax = axes[r,c]
            ax.annotate(lab, xy=(0,1), xycoords='axes fraction', va='top')
            if r==atl._nrows:
                ax.set_xlabel('storm length [days]')

            storm_length = np.array([self.sample({'weather_0':lab}) for _ in range(1000)])

            ax.hist(storm_length, bins=np.arange(1,30), cumulative=True, density=True)
            pdf_fitted,bins = np.histogram(storm_length, bins=np.arange(1,31), density=True)
            x = bins[1:] -0.5
            ax.plot(x,np.cumsum(pdf_fitted), c='r')
            ax.plot([1,np.median(storm_length),np.median(storm_length)],[0.5,0.5,0], c='r')
            ax.annotate(np.median(storm_length), xy=(np.median(storm_length),0.2), c='r', ha='left')
            ax.plot([1,np.percentile(storm_length,80),np.percentile(storm_length,80)],[0.8,0.8,0], c='r')
            ax.annotate(np.percentile(storm_length,80), xy=(np.percentile(storm_length,80),0.7), c='r', ha='left')

            ax.set_xlim(0,30)
            ax.set_ylim(0,1)
        plt.savefig(self._dir+'/sampled_storm_length_cumu.png', bbox_inches='tight'); plt.close()
