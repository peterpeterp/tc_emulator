
import numpy as np
import xarray as xr
import pandas as pd

import statsmodels.formula.api as smf

import seaborn as sns
import matplotlib.pyplot as plt


def sst_vs_wind_quantile_regression(tracks, quantiles = np.arange(0.1,1,0.1), plot_dir=None, sig=0.1, sst_var='sst'):

    # make quantile regression for sst
    mod = smf.quantreg('wind ~ '+sst_var, tracks)
    wind_quantiles = xr.DataArray(np.percentile(tracks.wind.values,quantiles*100), coords={'quantile':quantiles}, dims=['quantile'])

    wind_quR_params = xr.DataArray(0.,coords={'quantile':quantiles, 'param':['intercept','slope']}, dims=['quantile','param'])
    wind_quR_pvals = xr.DataArray(0.,coords={'quantile':quantiles, 'param':['intercept','slope']}, dims=['quantile','param'])
    for q in quantiles:
        res = mod.fit(q=q)
        if res.pvalues[1] < sig:
            wind_quR_params.loc[q,:] = res.params
            wind_quR_pvals.loc[q,:] = res.pvalues

    # plot results
    if plot_dir is not None:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(tracks[sst_var].min(), tracks[sst_var].max(), 0.1)
        for q in quantiles:
            ax.axhline(wind_quantiles.loc[q], color='gray')
            if wind_quR_params.loc[q,'slope'].values != 0:
                label = '%s perctl. %s kts/K pval: %s' %(round(q,2),round(float(wind_quR_params.loc[q,'slope'].values),2),round(float(wind_quR_pvals.loc[q,'slope']),2))
                if q == 0.5:
                    ax.plot(x, wind_quR_params.loc[q,'intercept'].values + wind_quR_params.loc[q,'slope'].values * x, linestyle='solid', color='r', label=label)
                else:
                    ax.plot(x, wind_quR_params.loc[q,'intercept'].values + wind_quR_params.loc[q,'slope'].values * x, linestyle='dotted', color='m', label=label)
        ax.scatter(tracks[sst_var], tracks.wind, alpha=.2)
        ax.set_xlabel('SST', fontsize=16)
        ax.set_ylabel('wind speed', fontsize=16);
        ax.legend()
        plt.savefig(plot_dir + sst_var + '_quR.png')

    return quantiles, wind_quR_params

def mod_tracks_sst(tracks, sst_mod, wind_quR_params, quantiles, sst_var='sst'):
    modified_tracks = tracks.copy()

    isabove = lambda p, a,b: np.cross(p-a, b-a) < 0

    sstMin,sstMax = float(tracks[sst_var].min()), float(tracks[sst_var].max())

    # shift wind and wind_before to match the corresponding sst_mod
    for var in ['wind','wind_before']:
        modified_tracks['qu_'+var] = 0.
        # assign data to different quantiles
        for i,qu in enumerate(quantiles[:-1]):
            a = np.array([sstMin, float(wind_quR_params.loc[qu,'intercept'] + sstMin * wind_quR_params.loc[qu,'slope'])])
            b = np.array([sstMax, float(wind_quR_params.loc[qu,'intercept'] + sstMax * wind_quR_params.loc[qu,'slope'])])
            # print(qu,modified_tracks.loc[:,[sst_var,var]].values)
            above = np.array([isabove(p,a,b) for p in list(modified_tracks.loc[:,[sst_var,var]].values)])

            a = np.array([sstMin, float(wind_quR_params.loc[:,'intercept'][i+1] + sstMin * wind_quR_params.loc[:,'slope'][i+1])])
            b = np.array([sstMax, float(wind_quR_params.loc[:,'intercept'][i+1] + sstMax * wind_quR_params.loc[:,'slope'][i+1])])
            above_next = np.array([isabove(p,a,b) for p in list(modified_tracks.loc[:,[sst_var,var]].values)])

            modified_tracks.loc[above & (above_next == False),['qu_'+var]] = i

        a = np.array([sstMin, float(wind_quR_params.loc[:,'intercept'][i+1] + sstMin * wind_quR_params.loc[:,'slope'][i+1])])
        b = np.array([sstMax, float(wind_quR_params.loc[:,'intercept'][i+1] + sstMax * wind_quR_params.loc[:,'slope'][i+1])])
        modified_tracks.loc[np.array([isabove(p,a,b) for p in list(modified_tracks.loc[:,[sst_var,var]].values)]), ['qu_'+var]] = i + 1

        for i,qu in enumerate(quantiles):
            sst_diff = sst_mod - modified_tracks.loc[modified_tracks['qu_'+var] == i, sst_var]
            modified_tracks.loc[modified_tracks['qu_'+var] == i, var] += sst_diff * wind_quR_params.loc[:,'slope'].values[i]

    return modified_tracks

def get_weather_distance(tracks, atl):
    # get pairwise distances between weather patterns
    weatherDistance = {}
    for lab in atl._grid_labels:
        tmp = np.sum((atl._axes_grid - atl._axes_grid[lab])**2, axis=1)
        weatherDistance[lab] = tmp

    # write them into a dataframe
    weather_dis = pd.DataFrame()
    weather_dis['weather_0'] = tracks['weather_0']
    for lab in atl._grid_labels:
        weather_dis[lab] = np.nan
        for lab2 in atl._grid_labels:
            weather_dis.loc[tracks.weather_0 == lab2, lab] = weatherDistance[lab][lab2]

    # calculate STD of differences for normalization
    weather_points = [atl._axes_grid[int(l)] for l in tracks['weather_0'].values]
    mean_weather_point = np.mean(np.array(weather_points), axis=0)
    weather_point_std = (np.sum(np.array([(p-mean_weather_point)**2 for p in weather_points])) / len(weather_points) )**0.5

    # normalize
    for lab in atl._grid_labels:
        weather_dis[lab] = weather_dis[lab] / weather_point_std # np.std(weather_dis['weather_0'])

    return weather_dis
