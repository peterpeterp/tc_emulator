# basics
import sys, os,pickle, inspect, textwrap, importlib, glob, itertools, inspect, resource, time, gc
from datetime import datetime, date, timedelta
import numpy as np
import xarray as xr
import pandas as pd
from collections import Counter
import collections

# for plotting
import seaborn as sns
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
from sklearn.metrics import brier_score_loss

import pymannkendall as mk

import warnings
warnings.filterwarnings('ignore')

import scipy
from scipy import stats

from addons.kl import KLdivergence
import addons.ndtest as ndtest

from tqdm import tqdm

from multiprocess import Pool


class storm_emulator(object):
    def __init__(self, dir, tag, emulate_season_function):
        self._dir = dir
        self._dir_sim = dir + '/' + tag + '/sim/'
        self._dir_plot = dir + '/' + tag +'/'
        os.system('mkdir -p '+self._dir_sim)
        self._tag = tag

        self.emulate_season = emulate_season_function

        self._validation_file = self._dir_plot + 'validation.pkl'
        if os.path.isfile(self._validation_file):
            self._validation = pickle.load(open(self._validation_file, 'rb'))
        else:
            self._validation = {}

    def save_validation(self):
        pickle.dump(self._validation, open(self._validation_file, 'wb'))

    def emulate_seasons(self, genesis_obj, wind_obj, end_obj, years, N=1000, fileName_addOn='', overwrite=False):
        exec("import %s; importlib.reload(%s); from %s import emulate_season_function" % tuple(['components.Emu.'+'Emu0']*3))

        self._N = N
        self._seasons = {}
        for year in years:
            fileName = self._dir_sim+'/' + self._tag+'_'+str(year)+'_N'+str(N)+fileName_addOn+'.pkl'
            if os.path.isfile(fileName) and overwrite==False:
                with open(fileName, 'rb') as infile:
                    self._seasons[year] = pickle.load(infile)
            else:
                tmp = self._weather_sst.loc[(self._weather_sst.year == year)]
                with Pool(5) as pool:
                    self._seasons[year] = list(
                        tqdm(pool.starmap(self.emulate_season, zip([tmp]*N,[genesis_obj]*N,[wind_obj]*N,[end_obj]*N))))
                # self._seasons[year]  = [self.emulate_season(tmp, genesis_obj, wind_obj, end_obj) for i in range(N)]
                gc.collect()

                with open(fileName, 'wb') as outfile:
                    pickle.dump(self._seasons[year], outfile)

    def emulate_seasons_serial(self, genesis_obj, wind_obj, end_obj, years, N=1000, fileName_addOn='', overwrite=False):
        self._N = N
        self._seasons = {}
        for year in years:
            fileName = self._dir_sim+'/' + self._tag+'_'+str(year)+'_N'+str(N)+fileName_addOn+'.pkl'
            if os.path.isfile(fileName) and overwrite==False:
                with open(fileName, 'rb') as infile:
                    self._seasons[year] = pickle.load(infile)
            else:
                tmp = self._weather_sst.loc[(self._weather_sst.year == year)]
                self._seasons[year]  = [self.emulate_season(tmp, genesis_obj, wind_obj, end_obj) for i in range(N)]

                with open(fileName, 'wb') as outfile:
                    pickle.dump(self._seasons[year], outfile)

                gc.collect()

    def get_simu_tracks(self, fileName_addOn='', fileName=None, overwrite=False):
        if fileName is None:
            fileName = self._dir_sim+'/'+self._tag+'_tracks_N'+str(self._N)+fileName_addOn+'.csv'

        if os.path.isfile(fileName) and overwrite==False:
            self._simu_tracks = pd.read_csv(fileName)

        else:
            self._simu_tracks = pd.DataFrame()
            for year,seasons in self._seasons.items():
                print(year)
                tmp = self._weather_sst.loc[self._weather_sst.year == year]
                for season in seasons:
                    for d0,storm in season.items():
                        for d,w in enumerate(storm):
                            d_in_seas = d0+d+3
                            self._simu_tracks = self._simu_tracks.append({'year':year, 'day_in_season':d_in_seas, 'wind':w,\
                                                                          'sst':float(tmp.loc[tmp.day_in_season==d_in_seas,'sst']),\
                                                                          'weather_0':float(tmp.loc[tmp.day_in_season==d_in_seas,'weather_0'])\
                                                                          }, ignore_index=True)
            self._simu_tracks.to_csv(fileName)

    def get_other_stats_for_tracks(self, tracks):
        self._simu_tracks['ACE'] = self._simu_tracks['wind'] ** 2 / 10000 * 4
        self._simu_tracks['Hur'] = self._simu_tracks['wind'] >= 64
        self._simu_tracks['MajHur'] = self._simu_tracks['wind'] >= 96

        self._tracks = tracks
        self._tracks['Hur'] = self._tracks['wind'] >= 64
        self._tracks['MajHur'] = self._tracks['wind'] >= 96

    def get_stats_seasonal_simu(self, fileName_addOn='', fileName=None, overwrite=False):
        if fileName is None:
            fileName = self._dir_sim+'/'+self._tag+'_stats_N'+str(self._N)+fileName_addOn+'.pkl'

        if os.path.isfile(fileName) and overwrite==False:
            with open(fileName, 'rb') as infile:
                self._simu = pickle.load(infile)

        else:
            years = list(self._seasons.keys())
            self._simu = {}

            # genesis
            self._simu['genesis'] = xr.DataArray([[len(list(sea.keys())) for sea in sea_] for sea_ in self._seasons.values()], coords={'year':years, 'run':np.arange(self._N)}, dims=['year','run'])

            # storm days
            self._simu['storm_days'] = xr.DataArray([[np.sum([len(sto) for sto in storms.values()]) for storms in sea_] for sea_ in self._seasons.values()], coords={'year':years, 'run':np.arange(self._N)}, dims=['year','run'])

            # wind
            self._simu['wind'] = xr.DataArray([[np.sum([np.sum([d for d in sto]) for sto in storms.values()]) for storms in sea_] for sea_ in self._seasons.values()], coords={'year':years, 'run':np.arange(self._N)}, dims=['year','run'])

            # Hur
            self._simu['Hur'] = xr.DataArray([[np.sum([np.max([d for d in sto]) >= 64 for sto in storms.values()]) for storms in sea_] for sea_ in self._seasons.values()], coords={'year':years, 'run':np.arange(self._N)}, dims=['year','run'])

            # MajHur
            self._simu['MajHur'] = xr.DataArray([[np.sum([np.max([d for d in sto]) >= 96 for sto in storms.values()]) for storms in sea_] for sea_ in self._seasons.values()], coords={'year':years, 'run':np.arange(self._N)}, dims=['year','run'])

            # ACE
            self._simu['ACE'] = xr.DataArray([[np.sum([np.sum([d**2 / 10000 * 4 for d in sto]) for sto in storms.values()]) for storms in sea_] for sea_ in self._seasons.values()], coords={'year':years, 'run':np.arange(self._N)}, dims=['year','run'])

            with open(fileName, 'wb') as outfile:
                pickle.dump(self._simu, outfile)

    def get_stats_seasonal_obs(self, tracks, years=None):
        if years is None:
            years = list(self._seasons.keys())

        self._obs = {}
        # genesis
        self._obs['genesis'] = xr.DataArray([tracks.loc[(tracks.year == year),'genesis'].sum() for year in years], coords={'year':years}, dims=['year'])

        # storm days
        self._obs['storm_days'] = xr.DataArray([np.sum(tracks.loc[(tracks.year == year),'wind']>0) for year in years], coords={'year':years}, dims=['year'])

        # wind
        self._obs['wind'] = xr.DataArray([tracks.loc[(tracks.year == year),'wind'].sum() for year in years], coords={'year':years}, dims=['year'])

        # ACE
        self._obs['ACE'] = xr.DataArray([tracks.loc[(tracks.year == year),'ACE'].sum() for year in years], coords={'year':years}, dims=['year'])

        # Hur
        self._obs['Hur'] = xr.DataArray(np.nan, coords={'year':years}, dims=['year'])
        for year in years:
            tmp = []
            for storm in np.unique(tracks.loc[(tracks.year == year),'storm']):
                tmp.append(np.nanmax(tracks.loc[(tracks.storm == storm),'wind']) >= 64)
            self._obs['Hur'].loc[year] = np.sum(tmp)

        # MajHur
        self._obs['MajHur'] = xr.DataArray(np.nan, coords={'year':years}, dims=['year'])
        for year in years:
            tmp = []
            for storm in np.unique(tracks.loc[(tracks.year == year),'storm']):
                tmp.append(np.nanmax(tracks.loc[(tracks.storm == storm),'wind']) >= 96)
            self._obs['MajHur'].loc[year] = np.sum(tmp)


#% evaluation and validation
    def sig_level(self,x):
        if x < 0.01:
            return '***'
        if x < 0.05:
            return '**'
        if x < 0.1:
            return '*'

        # if x >= 0.1:
        #     return ''
        return ''


    def vali_year_to_year_variability(self, indicator, show_legend=False, ax=None, version='mean'):
        '''
        check year to year variability
        plot time series
        save pearson correaltion
        '''
        if indicator not in self._validation.keys():
            self._validation[indicator] = {}

        if ax is None:
            plt.close('all')
            fig,ax = plt.subplots(nrows=1, figsize=(4,3))
            save = True
        else:
            save = False

        for set_ in self._sets:
            for years in set_['years']:
                tmp = self._simu[indicator].sel(year=years)
                ax.fill_between(years, np.nanpercentile(tmp,17,axis=1), np.nanpercentile(tmp,83,axis=1), alpha=0.5, color=set_['color'])
                ax.fill_between(years, np.nanpercentile(tmp,2.5,axis=1), np.nanpercentile(tmp,97.5,axis=1), alpha=0.5, color=set_['color'])
                if version == 'mean':
                    ax.plot(years, np.nanmean(tmp,axis=1), color=set_['color'])
                if version == 'median':
                    ax.plot(years, np.nanpercentile(tmp,50,axis=1), color=set_['color'])
                # ax.plot(years, np.nanmean(tmp,axis=1), color=set_['color'])
            years =  np.array([yr for l in set_['years'] for yr in l])
            tmp = self._simu[indicator].sel(year=years)

            for stat,name in zip([scipy.stats.pearsonr,scipy.stats.spearmanr],['pearson','spearman']):
                for aggr, aggr_name in zip([np.median,np.mean],['median','mean']):
                    corr,pval = stat(self._obs[indicator].sel(year=years), aggr(tmp,axis=1))
                    self._validation[indicator][name+'_'+aggr_name] = {'coef':corr, 'pval':pval}
                self._validation[indicator][name+'_runs'] = np.array([stat(self._obs[indicator].sel(year=years), tmp.loc[:,r]) for r in tmp.run.values])
            corr,pval = self._validation[indicator]['pearson_mean'].values()
            ax.plot([-99],[0], color=set_['color'], label='%s corr %s%s' %(set_['label'], round(corr,2), self.sig_level(pval)))

        years = self._obs[indicator].year
        ax.plot(years, self._obs[indicator], color='k', label='observed')
        ax.set_xticks(np.arange(years.min(),years.max(),5))
        ax.set_xticklabels(np.arange(years.min(),years.max(),5))
        ax.set_xlim(years.min(),years.max())
        ax.set_ylabel(self._indicator_dict[indicator])
        ax.set_xlabel('season')
        if show_legend:
            ax.legend(loc = 'upper left')
        self.save_validation()
        if save:
            out_file = self._dir_plot+'/seasonal_N'+str(self._N)+'_'+indicator+'.png'
            plt.savefig(out_file, transparent=True, dpi=200, bbox_inches='tight')
            return fig, ax, out_file
        else:
            return ax

    def vali_residuals_IND_vs_SST(self, indicator, SST, version='mean'):
        if indicator not in self._validation.keys():
            self._validation[indicator] = {}

        plt.close('all')
        fig,ax = plt.subplots(nrows=1, figsize=(4,3))
        for set in self._sets:
            for years in set['years']:
                SSTs_ = SST.sel(year=years)
                order = np.argsort(SSTs_.values)
                residuals_ = self._simu[indicator].sel(year=years) - self._obs[indicator].sel(year=years)
                residuals_ = residuals_[order]
                SSTs_ = SSTs_[order]
                ax.fill_between(SSTs_, np.nanpercentile(residuals_,17,axis=1), np.nanpercentile(residuals_,83,axis=1), alpha=0.5, color=set['color'])
                ax.fill_between(SSTs_, np.nanpercentile(residuals_,2.5,axis=1), np.nanpercentile(residuals_,97.5,axis=1), alpha=0.5, color=set['color'])
                if version == 'mean':
                    ax.plot(SSTs_, np.nanmean(residuals_,axis=1), color=set['color'])
                if version == 'median':
                    ax.plot(SSTs_, np.nanpercentile(residuals_,50,axis=1), color=set['color'])
            years =  np.array([yr for l in set['years'] for yr in l])
            ax.axhline(y=0, color='k', zorder=0)

            residuals = pd.DataFrame()
            residuals['year'] = np.array([[yr]*self._N for yr in years]).flatten()
            residuals['y'] = np.nan
            residuals['sst'] = np.nan
            for year in years:
                residuals.loc[residuals.year == year, 'y'] = residuals_.loc[year].values.flatten()
                residuals.loc[residuals.year == year, 'sst'] = SSTs_.loc[year].values.flatten()[0]

            wls_model = sm.WLS(residuals.y.values, sm.add_constant(residuals.sst.values)).fit()
            self._validation[indicator]['trend_vs_SST'] = {'slope':wls_model.params[1], 'pval':wls_model.pvalues[1]}

            wls_model = sm.WLS(residuals_.median('run').values, sm.add_constant(SSTs_.values)).fit()
            self._validation[indicator]['trend_vs_SST_median'] = {'slope':wls_model.params[1], 'pval':wls_model.pvalues[1]}
            if version == 'median':
                ax.plot(SSTs_, SSTs_ * wls_model.params[1] + wls_model.params[0], 'r--')
             
            wls_model = sm.WLS(residuals_.mean('run').values, sm.add_constant(SSTs_.values)).fit()
            self._validation[indicator]['trend_vs_SST_mean'] = {'slope':wls_model.params[1], 'pval':wls_model.pvalues[1]}
            if version == 'mean':
                ax.plot(SSTs_, SSTs_ * wls_model.params[1] + wls_model.params[0], 'r--')
            
            res = mk.original_test(residuals_.median('run').values)
            self._validation[indicator]['MK_vs_SST_median'] = {'trend':res.trend, 'pval':res.p}
            res = mk.original_test(residuals_.mean('run').values)
            self._validation[indicator]['MK_vs_SST_mean'] = {'trend':res.trend, 'pval':res.p}
            
        years = self._obs[indicator].year
        ax.set_ylabel(self._indicator_dict[indicator])
        ax.set_xlabel('seasonally averaged SST in the MDR [°C]')
        # ax.legend(loc='lower right')
        out_file = self._dir_plot+'/seasonal_N'+str(self._N)+'_'+indicator+'_residuals_vs_SSTs.png'
        plt.savefig(out_file, transparent=True, dpi=200, bbox_inches='tight')
        self.save_validation()
        return fig, ax, out_file

    def vali_residuals_and_long_term_trend(self, indicator, version='mean'):
        if indicator not in self._validation.keys():
            self._validation[indicator] = {}

        plt.close('all')
        fig,ax = plt.subplots(nrows=1, figsize=(4,3))
        for set in self._sets:
            for years in set['years']:
                residuals_ = self._simu[indicator].sel(year=years) - self._obs[indicator].sel(year=years)
                ax.fill_between(years, np.nanpercentile(residuals_,17,axis=1), np.nanpercentile(residuals_,83,axis=1), alpha=0.5, color=set['color'])
                ax.fill_between(years, np.nanpercentile(residuals_,2.5,axis=1), np.nanpercentile(residuals_,97.5,axis=1), alpha=0.5, color=set['color'])
                if version == 'mean':
                    ax.plot(years, np.nanmean(residuals_,axis=1), color=set['color'])
                if version == 'median':
                    ax.plot(years, np.nanpercentile(residuals_,50,axis=1), color=set['color'])
            years =  np.array([yr for l in set['years'] for yr in l])
            ax.plot([-99],[0], color=set['color'])
            ax.axhline(y=0, color='k', zorder=0)

            residuals = pd.DataFrame()
            residuals['year'] = np.array([[yr]*self._N for yr in years]).flatten()
            residuals['y'] = np.nan
            for year in years:
                residuals.loc[residuals.year == year, 'y'] = residuals_.loc[year].values.flatten()

            wls_model = sm.WLS(residuals.y.values, sm.add_constant(residuals.year.values)).fit()
            self._validation[indicator]['trend'] = {'slope':wls_model.params[1], 'pval':wls_model.pvalues[1]}

            wls_model = sm.WLS(residuals_.median('run').values, sm.add_constant(residuals_.year.values)).fit()
            self._validation[indicator]['trend_median'] = {'slope':wls_model.params[1], 'pval':wls_model.pvalues[1]}
            if version == 'median':
                ax.plot(years, years * wls_model.params[1] + wls_model.params[0], 'r--')

            wls_model = sm.WLS(residuals_.mean('run').values, sm.add_constant(residuals_.year.values)).fit()
            self._validation[indicator]['trend_mean'] = {'slope':wls_model.params[1], 'pval':wls_model.pvalues[1]}
            if version == 'mean':
                ax.plot(years, years * wls_model.params[1] + wls_model.params[0], 'r--')
            
            res = mk.original_test(residuals_.median('run').values)
            self._validation[indicator]['MK_median'] = {'trend':res.trend, 'pval':res.p}
            res = mk.original_test(residuals_.mean('run').values)
            self._validation[indicator]['MK_mean'] = {'trend':res.trend, 'pval':res.p}            
            
        years = self._obs[indicator].year
        ax.set_xticks(np.arange(years.min(),years.max(),5))
        ax.set_xticklabels(np.arange(years.min(),years.max(),5))
        ax.set_xlim(years.min(),years.max())
        ax.set_ylabel(self._indicator_dict[indicator])
        ax.set_xlabel('season')
        out_file = self._dir_plot+'/seasonal_N'+str(self._N)+'_'+indicator+'_residuals.png'
        plt.savefig(out_file, transparent=True, dpi=200, bbox_inches='tight')
        self.save_validation()
        return fig, ax, out_file

    def vali_outliers(self, indicator, quantiles=[5,17,83,95], show_legend=False):
        '''
        check year to year variability
        plot time series
        save pearson correaltion
        '''
        if indicator not in self._validation.keys():
            self._validation[indicator] = {}

        self._validation[indicator]['outliers'] = {}

        plt.close('all')
        fig,ax = plt.subplots(nrows=1, figsize=(4,3))
        for q in quantiles:
            for set_ in self._sets:
                for years in set_['years']:
                    base = np.min([q, 100-q]) / len(years)
                    simu_quaD = np.nanpercentile(self._simu[indicator].sel(year=years), q, axis=1) - self._obs[indicator].values
                    if q > 50:
                        self._validation[indicator]['outliers'][q] = np.sum(simu_quaD < 0) / len(years) #- base
                    if q < 50:
                        self._validation[indicator]['outliers'][q] = np.sum(simu_quaD > 0) / len(years) #- base
                    plt.plot(self._obs[indicator].year, simu_quaD, label='%sth percentile' %(q))

        plt.axhline(0, color='k', linestyle='--')
        years = self._obs[indicator].year
        ax.set_xticks(np.arange(years.min(),years.max(),5))
        ax.set_xticklabels(np.arange(years.min(),years.max(),5))
        ax.set_xlim(years.min(),years.max())
        ax.set_ylabel(self._indicator_dict[indicator])
        ax.legend(loc='bottom right')
        out_file = self._dir_plot+'/seasonal_N'+str(self._N)+'_'+indicator+'_quantiles.png'
        plt.savefig(out_file, transparent=True, dpi=200, bbox_inches='tight')
        self.save_validation()
        return fig, ax, out_file

    def vali_RMSD(self, indicator):

        if indicator not in self._validation.keys():
            self._validation[indicator] = {}


        for set_ in self._sets:
            for years in set_['years']:
                rmss = (np.sum(np.array(self._simu[indicator].sel(year=years).median('run').values - self._obs[indicator].values)**2) / len(years) )**0.5
                self._validation[indicator]['RMSD'] = rmss

        self.save_validation()

    def vali_distr_tests(self, obs, simu, out_file, indicator='wind', nBins=30, bins=None):
        if indicator not in self._validation.keys():
            self._validation[indicator] = {}

        years = list(self._seasons.keys())


        stat, pval = stats.ks_2samp(simu, obs)
        self._validation[indicator]['KS'] = {'stat':stat, 'pval':pval}

        if bins is None:
            bins = np.linspace(np.min(np.concatenate((obs,simu))), np.max(np.concatenate((obs,simu))), nBins)

        probs_obs = np.histogram(obs, bins=bins, density=True)[0]
        probs_simu = np.histogram(simu, bins=bins, density=True)[0]

        # remove zeros and normailiye
        probs_obs += 0.000001
        probs_simu += 0.000001
        probs_obs /= probs_obs.sum()
        probs_simu /= probs_simu.sum()

        KL = 0
        for po,ps,i in zip(probs_obs,probs_simu,range(probs_obs.shape[0])):
            KL += po * np.log(po/ps)

        self._validation[indicator]['KL'] = {'stat':KL}
        self.save_validation()

        plt.close('all')
        fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3), sharey=True)
        ax.hist(obs, density=True, bins=bins, alpha=0.5, label='observed')
        ax.hist(simu, density=True, bins=bins, alpha=0.5, label='simulated')
        ax.legend(loc='center right')
        # ax.set_xlabel('wind speed [kts]')
        # out_file = self._dir_plot+'hist_'+indicator+'_N'+str(self._N)+'.png'
        plt.savefig(out_file, transparent=True, bbox_inches='tight')

        return fig, ax, out_file

    def vali_2D_distr_plot(self, obs, simu, ranges=None, bw_method=None, legend=True, nBins=100):
        x_name,y_name = obs.columns

        indicator = y_name+'_vs_'+x_name
        if indicator not in self._validation.keys():
            self._validation[indicator] = {}

        obs,simu = obs.values, simu.values

        if ranges is None:
            xs,ys = np.concatenate((obs[:,0],simu[:,0])), np.concatenate((obs[:,1],simu[:,1]))
            X, Y = np.meshgrid(np.linspace(np.min(xs),np.max(xs),nBins), np.linspace(np.min(ys),np.max(ys),nBins))
        else:
            X, Y = np.meshgrid(np.linspace(ranges[0],ranges[1],nBins), np.linspace(ranges[2],ranges[3],nBins))

        positions = np.vstack([X.ravel(), Y.ravel()])
        fig,ax= plt.subplots(nrows=1, ncols=1, figsize=(3,3))

        kernel = stats.gaussian_kde(obs.T, bw_method=bw_method)
        Z = np.reshape(kernel(positions).T, X.shape)
        im_obs = ax.contourf(X,Y,Z, cmap='Greys')

        kernel = stats.gaussian_kde(simu.T, bw_method=bw_method)
        Z = np.reshape(kernel(positions).T, X.shape)
        im_simu = ax.contour(X,Y,Z, cmap='plasma')

        ax.set_ylabel(self._indicator_dict[y_name])
        ax.set_xlabel(self._indicator_dict[x_name])

        # x = obs.values # + np.random.sample(obs.shape) * 0.001
        # y = simu.values # + np.random.sample(simu.shape) * 0.001
        # pval,ksd = ndtest.ks2d2s(x[:,0],x[:,1],y[:,0],y[:,1],extra=True)

        # ax.annotate('KS=%s pval=%s' %(round(ksd,2), round(pval,3)), xy=(0.95,0.95), xycoords='axes fraction', ha='right', va='top')

        if legend:
            cax = fig.add_axes([0.7, 0.2, 0.15, 0.05])
            cax.annotate('observed ', xy=(0,0.5), xycoords='axes fraction', ha='right')
            cb = matplotlib.colorbar.ColorbarBase(cax, orientation='horizontal', cmap=plt.cm.Greys, norm=matplotlib.colors.Normalize(0, 10), ticks=[])

            cax = fig.add_axes([0.7, 0.15, 0.15, 0.05])
            cax.annotate('simulated ', xy=(0,0.5), xycoords='axes fraction', ha='right')
            cb = matplotlib.colorbar.ColorbarBase(cax, orientation='horizontal', cmap=plt.cm.plasma, norm=matplotlib.colors.Normalize(0, 10), ticks=[])

        out_file = self._dir_plot+indicator+'_N'+str(self._N)+'.png'
        plt.savefig(out_file, bbox_inches='tight', dpi=200)

        self.save_validation()
        return fig, ax, im_simu, im_obs, out_file

    def vali_2D_distr_KL_divergence(self, obs, simu, ranges=None, legend=True, nBins=100):
        x_name,y_name = obs.columns

        indicator = y_name+'_vs_'+x_name
        if indicator not in self._validation.keys():
            self._validation[indicator] = {}

        obs,simu = obs.values, simu.values

        if ranges is None:
            xs,ys = np.concatenate((obs[:,0],simu[:,0])), np.concatenate((obs[:,1],simu[:,1]))
            X, Y = np.meshgrid(np.linspace(np.min(xs),np.max(xs),nBins), np.linspace(np.min(ys),np.max(ys),nBins))
        else:
            X, Y = np.meshgrid(ranges[0],ranges[1])


        # Kullback-Leibler statistic (2D)
        positions = np.vstack([X.ravel(), Y.ravel()])
        probs_obs , probs_simu, xx, yy = [],[],[],[]
        for y_l,y_h in zip(np.unique(Y)[:-1], np.unique(Y)[1:]):
            for x_l,x_h in zip(np.unique(X)[:-1], np.unique(X)[1:]):
                probs_obs.append( np.sum((obs[:,0] <= x_h) & (obs[:,0] > x_l) & (obs[:,1] <= y_h) & (obs[:,1] > y_l)) )
                probs_simu.append( np.sum((simu[:,0] <= x_h) & (simu[:,0] > x_l) & (simu[:,1] <= y_h) & (simu[:,1] > y_l)) )
                xx.append( np.mean([x_l,x_h]) )
                yy.append( np.mean([y_l,y_h]) )
        # remove zeros and normailiye
        probs_obs = np.array(probs_obs) + 0.000001
        probs_simu = np.array(probs_simu) + 0.000001
        probs_obs /= probs_obs.sum()
        probs_simu /= probs_simu.sum()

        KL = 0
        for po,ps,i in zip(probs_obs,probs_simu,range(probs_obs.shape[0])):
            KL += po * np.log(po/ps)

        self._validation[indicator]['KL'] = {'stat':KL}

        fig,ax= plt.subplots(nrows=1, ncols=1, figsize=(3,3))

        ax.annotate('KL=%s' %(round(KL,3)), xy=(0.95,0.95), xycoords='axes fraction', ha='right', va='top')

        for x,y,s1,s2 in zip(xx,yy,probs_obs,probs_simu):
            ax.plot(x,y, marker='o', markersize=s1*200, color='blue', fillstyle='right', markeredgecolor=(0,0,0,0))
            ax.plot(x,y, marker='o', markersize=s2*200, color='orange', fillstyle='left', markeredgecolor=(0,0,0,0))

        ax.set_ylabel(self._indicator_dict[y_name])
        ax.set_xlabel(self._indicator_dict[x_name])

        out_file = self._dir_plot+indicator+'_KL_N'+str(self._N)+'.png'
        plt.savefig(out_file, bbox_inches='tight', dpi=200)

        self.save_validation()
        return fig, ax, out_file

    def evaluate_skill(self):
        skill = {}
        for indicator in ['genesis','storm_days','wind','ACE']:
            skill[indicator] = {}
            for set in self._sets:
                name = set['label']
                years =  np.array([yr for l in set['years'] for yr in l])
                simu = self._simu[indicator].sel(year=years)
                obs = self._obs[indicator].sel(year=years)

                tmp = {'years':years}

                tmp['bias'] = simu.mean().values - obs.mean().values
                tmp['bias_rel'] = (simu.mean().values - obs.mean().values) / obs.mean().values

                tmp['pearson'] = scipy.stats.pearsonr(obs, simu.median('run'))
                tmp['spearman'] = scipy.stats.spearmanr(obs, simu.median('run'))

                tmp['bss'] = {}
                for qu in [17,33,50,66,83]:
                    bs = brier_score_loss(obs>np.percentile(obs,qu), (simu - np.percentile(simu,qu) > 0).mean('run'))
                    bsc = brier_score_loss(obs>np.percentile(obs,qu), [(100-qu)/100]*obs.shape[0])
                    tmp['bss'][qu] = (bsc - bs) / bsc

                tmp['taylor_stats'] = sm.taylor_statistics({'data':simu.median('run')},{'data':obs},'data')

                skill[indicator][name] = tmp

        with open(self._dir_plot+'/seasonal_skills_N'+str(self._N)+'.pkl', 'wb') as outfile:
            pickle.dump(skill, outfile)

        self._skill = skill

        sdev = np.array([tmp['taylor_stats']['sdev'][0]]+[vals['taylor_stats']['sdev'][1] for name,vals in skill[indicator].items()])
        crmsd = np.array([tmp['taylor_stats']['crmsd'][0]]+[vals['taylor_stats']['crmsd'][1] for name,vals in skill[indicator].items()])
        ccoef = np.array([tmp['taylor_stats']['ccoef'][0]]+[vals['taylor_stats']['ccoef'][1] for name,vals in skill[indicator].items()])
        name = np.array(['obs']+[name for name,vals in skill[indicator].items()])

        plt.close('all')
        sm.taylor_diagram(sdev,crmsd,ccoef, markercolor='r', alpha = 0.0, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0])
        sm.taylor_diagram(sdev,crmsd,ccoef, markerLabel=list(name), markerLegend='on', markercolor='b', overlay='on')
        plt.savefig(self._dir_plot+'taylor1.png')



#% counterfactual
    def simulate_sst_counterfactual(self, name, sst_shift, years, genesis_obj, wind_obj, end_obj, N=100, overwrite=False):
        self._N = N

        fileName = self._dir_sim+'/'+self._tag+'_stats_N'+str(self._N)+'_'+name+'.pkl'
        if os.path.isfile(fileName) and overwrite==False:
            with open(fileName, 'rb') as infile:
                self._simu = pickle.load(infile)

        else:
            self._seasons = {}
            for year in years:
                fileName = self._dir_sim+'/'+self._tag+'_'+str(year)+'_N'+str(self._N)+name+'.pkl'
                if os.path.isfile(fileName) and overwrite==False:
                    with open(fileName, 'rb') as infile:
                        self._seasons[year] = pickle.load(infile)
                else:
                    tmp = self._weather_sst.loc[(self._weather_sst.year == year)].copy()
                    tmp.sst += sst_shift
                    self._seasons[year]  = [self.emulate_season(tmp, genesis_obj, wind_obj, end_obj) for i in range(N)]

                    with open(fileName, 'wb') as outfile:
                        pickle.dump(self._seasons[year], outfile)

            self.get_stats_seasonal_simu(name, overwrite=overwrite)

            if hasattr(self, '_counterFacts') == False:
                self._counterFacts = {}

            self._counterFacts[name] = self._simu

    def plot_counterfactual_compare(self, counter_dict, tag='test'):

        plt.close('all')
        with PdfPages(self._dir+'/'+self._tag+'/counter_'+tag+'_pdf_smoo.pdf') as pdf:
            for indicator in self._obs.keys():
                fig,ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
                for name,details in counter_dict.items():
                    X = np.linspace(self._counterFacts[name][indicator].min(),self._counterFacts[name][indicator].max(),1000)
                    bw = (self._counterFacts[name][indicator].max() - self._counterFacts[name][indicator].min()) / 20
                    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(self._counterFacts[name][indicator].mean('year').values.reshape(-1, 1))
                    log_dens = kde.score_samples(X.reshape(-1, 1))
                    y = np.exp(log_dens) / np.exp(log_dens).sum()
                    y = y / y.sum()
                    ax.plot(X,y, color=details['color'], linestyle=details['lsty'], label=details['label'])

                ax.axvline(x=self._obs[indicator].mean('year'), color='k', label='%s observed' %(tag))
                ax.legend(fontsize=6)
                ax.set_xlabel(indicator)
                pdf.savefig(bbox_inches='tight'); plt.close()

    def plot_FAR_curves(self, counter, ref, threshold_dict={'genesis':10}, N=1000, tag='test'):

        X = np.linspace(0,300,1000)
        plt.close('all')
        with PdfPages(self._dir+'/FAR_'+indicator+'_'+'_FAR.pdf') as pdf:
            fig,ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

            for indicator,threshold in threshold_dict.items():
                FAR = []
                for i in range(N):
                    probRef = np.sum(np.random.choice(ref[indicator].values.flatten(), size=int(0.5*N), replace=True) >= threshold) / (0.5*N)
                    probCounter = np.sum(np.random.choice(counter[indicator].values.flatten(), size=int(0.5*N), replace=True) >= threshold) / (0.5*N)
                    FAR.append(1 - probCounter / probRef)
                FAR = np.array(FAR)
                ax.hist(FAR, bins=np.linspace(0,1,20), density=True)
                ax.axvline(x=np.median(FAR), color='k', label='median')
                ax.legend()
                ax.set_xlabel('Fractional Attributable Risk (FAR)')
                ax.set_ylabel('Probability')
                pdf.savefig(bbox_inches='tight'); plt.close()

    def old_cunftion(self):

        plt.close('all')
        with PdfPages(self._dir+'/'+self._tag+'/counter_'+tag+'_N'+str(N)+'_pdf.pdf') as pdf:
            for indicator in self._obs.keys():
                X = np.linspace(sst_dict[name]['simu'][indicator].min(),sst_dict[name]['simu'][indicator].max(),1000)
                fig,ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
                for name,details in sst_dict.items():
                    if np.all(np.diff(np.unique(np.abs(sst_dict[name]['simu'][indicator].mean('year'))),1) % 1 == 0) and np.unique(sst_dict[name]['simu'][indicator].mean('year')).shape[0] < 20:
                        bins = np.arange(sst_dict[name]['simu'][indicator].mean('year').min(), sst_dict[name]['simu'][indicator].mean('year').max()+1, 1)
                    else:
                        bins = 20
                    pdf_,bins = np.histogram(sst_dict[name]['simu'][indicator].mean('year'), bins=bins, density=True)
                    pdf_ /= pdf_.sum()
                    bins = bins[:-1] + np.diff(bins,1)[0] * 0.5
                    ax.plot(bins,pdf_,color=details['color'],label=name, alpha=1)

                ax.axvline(x=self._obs[indicator].mean('year'), color='k', label='%s observed' %(tag))
                ax.legend()
                ax.set_xlabel(indicator)
                pdf.savefig(bbox_inches='tight'); plt.close()



        plt.close('all')
        with PdfPages(self._dir+'/'+self._tag+'/counter_'+tag+'_N'+str(N)+'_cdf.pdf') as pdf:
            for indicator in self._obs.keys():
                X = np.linspace(sst_dict[name]['simu'][indicator].min(),sst_dict[name]['simu'][indicator].max(),1000)
                fig,ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
                for name,details in sst_dict.items():
                    if np.all(np.diff(np.unique(np.abs(sst_dict[name]['simu'][indicator].mean('year'))),1) % 1 == 0) and np.unique(sst_dict[name]['simu'][indicator].mean('year')).shape[0] < 20:
                        bins = np.arange(sst_dict[name]['simu'][indicator].mean('year').min(), sst_dict[name]['simu'][indicator].mean('year').max()+1, 1)
                    else:
                        bins = 20
                    pdf_,bins = np.histogram(sst_dict[name]['simu'][indicator].mean('year'), bins=bins, density=True)
                    pdf_ /= pdf_.sum()
                    bins = bins[:-1] + np.diff(bins,1)[0] * 0.5
                    ax.plot(bins,np.cumsum(pdf_) ,color=details['color'],label=name, alpha=1)

                ax.axvline(x=self._obs[indicator].mean('year'), color='k', label='%s observed' %(tag))
                ax.legend()
                ax.set_xlabel(indicator)
                pdf.savefig(bbox_inches='tight'); plt.close()


        plt.close('all')
        with PdfPages(self._dir+'/'+self._tag+'/counter_'+tag+'_N'+str(N)+'_cdf_smoo.pdf') as pdf:
            for indicator in self._obs.keys():
                X = np.linspace(sst_dict[name]['simu'][indicator].min(),sst_dict[name]['simu'][indicator].max(),1000)
                fig,ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
                for name,details in sst_dict.items():
                    bw = (sst_dict[name]['simu'][indicator].max() - sst_dict[name]['simu'][indicator].min()) / 20
                    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(sst_dict[name]['simu'][indicator].mean('year').values.reshape(-1, 1))
                    log_dens = kde.score_samples(X.reshape(-1, 1))
                    y = np.exp(log_dens) / np.exp(log_dens).sum()
                    y = y / y.sum()
                    ax.plot(X,np.cumsum(y),color=details['color'],label=name)

                ax.axvline(x=self._obs[indicator].mean('year'), color='k', label='%s observed' %(tag))
                ax.legend()
                ax.set_xlabel(indicator)
                pdf.savefig(bbox_inches='tight'); plt.close()


'''


    def plot_seasonal_evolution(self,year):
        plt.close('all')
        with PdfPages(self._dir+'/seasonal_evolution_'+self._tag+'_N'+str(N)+'.pdf') as pdf:
            for storms in self._seasons[year]:
                for storm_id, winds in storms.items():
                    plt.plot(winds)
                pdf.savefig(bbox_inches='tight'); plt.close()

        plt.close('all')
        wind = {i:[] for i in range(30)}
        for storms in self._seasons[year]:
            for storm_id, winds in storms.items():
                plt.plot(winds, color='gray', linewidth=5, alpha=0.01)
                for i,w in enumerate(winds):
                    wind[i].append(w)
        plt.savefig(self._dir+'/seasonal_evolution_'+self._tag+'_N'+str(N)+'_v1.png')

        perc = {q:[] for q in [0,10,33,50,66,90,100]}
        for i in range(22):
            qus = np.nanpercentile(wind[i], [0,10,33,50,66,90,100])
            for j,q in enumerate([0,10,33,50,66,90,100]):
                perc[q].append(qus[j])
        for qu in [0,10,33,50,66,90,100]:
            plt.plot(perc[qu])
        plt.savefig(self._dir+'/seasonal_evolution_'+self._tag+'_N'+str(N)+'.png')

    def plot_year_to_year(self, indicator):
        plt.close('all')
        fig,ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        for set in self._sets:
            for years in set['years']:
                tmp = self._simu[indicator].sel(year=years)
                ax.fill_between(years, np.nanpercentile(tmp,17,axis=1), np.nanpercentile(tmp,83,axis=1), alpha=0.5, color=set['color'])
                ax.fill_between(years, np.nanpercentile(tmp,0,axis=1), np.nanpercentile(tmp,100,axis=1), alpha=0.5, color=set['color'])
                ax.plot(years, np.nanpercentile(tmp,50,axis=1), color=set['color'])
                ax.plot(years, np.nanmean(tmp,axis=1), color=set['color'])
            years =  np.array([yr for l in set['years'] for yr in l])
            corr,pval = scipy.stats.pearsonr(self._obs[indicator].sel(year=years), self._simu[indicator].sel(year=years).median('run'))
            corr2,pval2 = scipy.stats.pearsonr(self._obs[indicator].sel(year=years), self._simu[indicator].sel(year=years).mean('run'))
            sig_level = lambda x : '**' if x < 0.05 else '' if x > 0.1 else '*'
            ax.plot([-99],[0], color=set['color'], label='%s corr %s%s %s%s' %(set['label'], round(corr,2), sig_level(pval) , round(corr2,2), sig_level(pval2)))

        years = self._obs[indicator].year
        ax.plot(years, self._obs[indicator], color='k', label='observed')
        ax.set_xticks(np.arange(years.min(),years.max(),5))
        ax.set_xticklabels(np.arange(years.min(),years.max(),5))
        ax.set_xlim(years.min(),years.max())
        ax.set_ylabel(self._indicator_dict[indicator])
        ax.legend()
        plt.savefig(self._dir_plot+'/seasonal_N'+str(self._N)+'_'+indicator+'.png', transparent=True, bbox_inches='tight'); plt.close()

    def plot_year_to_year_residuals(self, indicator):
        plt.close('all')
        fig,ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        for set in self._sets:
            for years in set['years']:
                tmp = self._simu[indicator].sel(year=years) - self._obs[indicator].sel(year=years)
                ax.fill_between(years, np.nanpercentile(tmp,17,axis=1), np.nanpercentile(tmp,83,axis=1), alpha=0.5, color=set['color'])
                ax.fill_between(years, np.nanpercentile(tmp,0,axis=1), np.nanpercentile(tmp,100,axis=1), alpha=0.5, color=set['color'])
                ax.plot(years, np.nanpercentile(tmp,50,axis=1), color=set['color'])
            years =  np.array([yr for l in set['years'] for yr in l])
            corr,pval = scipy.stats.pearsonr(self._obs[indicator].sel(year=years), self._simu[indicator].sel(year=years).median('run'))
            sig_level = lambda x : '**' if x < 0.05 else '' if x > 0.1 else '*'
            ax.plot([-99],[0], color=set['color'], label='%s corr %s%s' %(set['label'], round(corr,2), sig_level(pval)))

        ax.axhline(0, color='k')
        years = self._obs[indicator].year
        ax.set_xticks(np.arange(years.min(),years.max(),5))
        ax.set_xticklabels(np.arange(years.min(),years.max(),5))
        ax.set_xlim(years.min(),years.max())
        ax.set_ylabel(self._indicator_dict[indicator])
        ax.legend()
        plt.savefig(self._dir_plot+'/seasonal_N'+str(self._N)+'_'+indicator+'_residuals.png', transparent=True, bbox_inches='tight'); plt.close()

    def evaluate_skill(self):
        skill = {}
        for indicator in ['genesis','storm_days','wind','ACE']:
            skill[indicator] = {}
            for set in self._sets:
                name = set['label']
                years =  np.array([yr for l in set['years'] for yr in l])
                simu = self._simu[indicator].sel(year=years)
                obs = self._obs[indicator].sel(year=years)

                tmp = {'years':years}

                tmp['bias'] = simu.mean().values - obs.mean().values
                tmp['bias_rel'] = (simu.mean().values - obs.mean().values) / obs.mean().values

                tmp['pearson'] = scipy.stats.pearsonr(obs, simu.median('run'))
                tmp['spearman'] = scipy.stats.spearmanr(obs, simu.median('run'))

                tmp['bss'] = {}
                for qu in [17,33,50,66,83]:
                    bs = brier_score_loss(obs>np.percentile(obs,qu), (simu - np.percentile(simu,qu) > 0).mean('run'))
                    bsc = brier_score_loss(obs>np.percentile(obs,qu), [(100-qu)/100]*obs.shape[0])
                    tmp['bss'][qu] = (bsc - bs) / bsc

                tmp['taylor_stats'] = sm.taylor_statistics({'data':simu.median('run')},{'data':obs},'data')

                skill[indicator][name] = tmp

        with open(self._dir_plot+'/seasonal_skills_N'+str(self._N)+'.pkl', 'wb') as outfile:
            pickle.dump(skill, outfile)

        self._skill = skill

        sdev = np.array([tmp['taylor_stats']['sdev'][0]]+[vals['taylor_stats']['sdev'][1] for name,vals in skill[indicator].items()])
        crmsd = np.array([tmp['taylor_stats']['crmsd'][0]]+[vals['taylor_stats']['crmsd'][1] for name,vals in skill[indicator].items()])
        ccoef = np.array([tmp['taylor_stats']['ccoef'][0]]+[vals['taylor_stats']['ccoef'][1] for name,vals in skill[indicator].items()])
        name = np.array(['obs']+[name for name,vals in skill[indicator].items()])

        plt.close('all')
        sm.taylor_diagram(sdev,crmsd,ccoef, markercolor='r', alpha = 0.0, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0])
        sm.taylor_diagram(sdev,crmsd,ccoef, markerLabel=list(name), markerLegend='on', markercolor='b', overlay='on')
        plt.savefig(self._dir_plot+'taylor1.png')

    def plot_maxWind_vs_length(self, tracks):
        seasons = [winds for season in self._seasons.values() for storms in season for winds in storms.values()]
        years = list(self._seasons.keys())

        plt.close('all')
        fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4), sharey=True)
        storms = [[np.float(w) for w in tracks.loc[(tracks.storm==storm), 'wind']] for storm in np.unique(tracks.loc[np.isin(tracks.year,years),'storm'])]
        xy = np.array([[len(winds),max(winds)] for winds in storms])
        kernel = stats.gaussian_kde(xy.T)
        X, Y = np.mgrid[0:20:100j, 0:180:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        axes[0].contourf(X,Y,Z, cmap='plasma')
        axes[2].contourf(X,Y,Z, cmap='Reds', label='observed')
        # axes[0].scatter(xy[:,0],xy[:,1], marker='.', color='g', alpha=0.5)
        axes[0].set_title('observed')

        xy = np.array([[len(winds),max(winds)] for winds in seasons])
        kernel = stats.gaussian_kde(xy.T)
        Z = np.reshape(kernel(positions).T, X.shape)
        axes[1].contourf(X,Y,Z, cmap='plasma')
        axes[2].contour(X,Y,Z, cmap='Blues', label='simulated')
        axes[1].set_title('simulated')
        # axes[1].scatter(xy[:,0],xy[:,1], marker='.', color='g', alpha=0.5)

        for ax in axes:
            ax.set_xlabel('days')
        axes[0].set_ylabel('maximum wind speed [kts]')
        axes[2].legend()
        plt.savefig(self._dir_plot+'maxWind_vs_stormLength_N'+str(self._N)+'.png', transparent=True, bbox_inches='tight'); plt.close()

    def plot_hist_wind(self, tracks):
        seasons = [winds for season in self._seasons.values() for storms in season for winds in storms.values()]
        years = list(self._seasons.keys())

        plt.close('all')
        fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4), sharey=True)
        storms = [[np.float(w) for w in tracks.loc[(tracks.storm==storm), 'wind']] for storm in np.unique(tracks.loc[np.isin(tracks.year,years),'storm'])]
        winds = np.array([wind for winds in storms for wind in winds])
        axes[0].hist(winds, density=True, alpha=0.5)
        axes[2].hist(winds, density=True, alpha=0.5)

        winds = np.array([wind for winds in seasons for wind in winds])
        axes[1].hist(winds, density=True, alpha=0.5)
        axes[2].hist(winds, density=True, alpha=0.5)

        for ax in axes:
            ax.set_xlabel('wind speed [kts]')
        plt.savefig(self._dir_plot+'windHist_N'+str(self._N)+'.png', transparent=True, bbox_inches='tight'); plt.close()

    def plot_hist_storm_length(self, tracks):
        years = list(self._seasons.keys())

        plt.close('all')
        fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4), sharey=True)
        storms = tracks.loc[(tracks.genesis==1) & np.isfinite(tracks.weather_0), 'storm']
        storm_length = np.array([tracks.loc[(tracks.storm == sto)].shape[0] for sto in storms])
        pdf,bins = np.histogram(storm_length, bins=np.arange(0.5,20.5,1), density=True)
        axes[0].plot(bins[:-1]+0.5, pdf)
        axes[2].plot(bins[:-1]+0.5, pdf)

        storm_length = [sto.shape[0] for season in self._seasons.values() for storms in season for sto in storms.values()]
        pdf,bins = np.histogram(storm_length, bins=np.arange(0.5,20.5,1), density=True)
        axes[1].plot(bins[:-1]+0.5, pdf)
        axes[2].plot(bins[:-1]+0.5, pdf)

        for ax in axes:
            ax.set_xlabel('storm length')
        plt.savefig(self._dir_plot+'hist_stormLength_N'+str(self._N)+'.png', transparent=True, bbox_inches='tight'); plt.close()

    def plot_windSpeed_vs_day(self, tracks):
        seasons = [winds for season in self._seasons.values() for storms in season for winds in storms.values()]
        years = list(self._seasons.keys())

        plt.close('all')
        fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4), sharey=True)
        storms = [[np.float(w) for w in tracks.loc[(tracks.storm==storm), 'wind']] for storm in np.unique(tracks.loc[np.isin(tracks.year,years),'storm'])]
        xy = np.array([[i,wind] for winds in storms for i,wind in enumerate(winds[1:])])
        axes[0].hist2d(xy[:,0], xy[:,1], bins=20, cmap='plasma')#, norm=matplotlib.colors.LogNorm())

        xy = np.array([[i,wind] for winds in seasons for i,wind in enumerate(winds[1:])])
        axes[1].hist2d(xy[:,0], xy[:,1], bins=20, cmap='plasma')#, norm=matplotlib.colors.LogNorm())


        for ax in axes:
            ax.set_xlabel('storm day')
        axes[0].set_ylabel('wind speed [kts]')
        plt.savefig(self._dir_plot+'/windSpeed_vs_windDay_N'+str(self._N)+'.png', transparent=True, bbox_inches='tight'); plt.close()

    def plot_windSpeed_vs_windDay(self, tracks):
        seasons = [winds for season in self._seasons.values() for storms in season for winds in storms.values()]
        years = list(self._seasons.keys())

        plt.close('all')
        fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4), sharey=True)
        storms = [[np.float(w) for w in tracks.loc[(tracks.storm==storm), 'wind']] for storm in np.unique(tracks.loc[np.isin(tracks.year,years),'storm'])]
        xy = np.array([[i+1,wind] for winds in storms for i,wind in enumerate(winds[1:])])
        kernel = stats.gaussian_kde(xy.T, bw_method=0.5)
        X, Y = np.mgrid[1:10:100j, 0:120:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        axes[0].contourf(X,Y,Z, cmap='plasma')
        axes[2].contourf(X,Y,Z, cmap='Reds')
        axes[0].set_title('observed')

        xy = np.array([[i,wind] for winds in seasons for i,wind in enumerate(winds[1:])])
        kernel = stats.gaussian_kde(xy.T, bw_method=0.5)
        Z = np.reshape(kernel(positions).T, X.shape)
        axes[1].contourf(X,Y,Z, cmap='plasma')
        axes[2].contour(X,Y,Z, cmap='Blues')
        axes[1].set_title('simulated')

        for ax in axes:
            ax.set_xlabel('storm day')
        axes[0].set_ylabel('wind speed [kts]')
        plt.savefig(self._dir_plot+'/windSpeed_vs_windDay_N'+str(self._N)+'.png', transparent=True, bbox_inches='tight'); plt.close()


'''

#
