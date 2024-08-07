# basics
import sys, os,pickle, inspect, textwrap, importlib, glob, itertools, inspect, resource, time
from datetime import datetime, date, timedelta
import numpy as np
import xarray as xr
import pandas as pd

def emulate_season_function(tmp, genesis_obj, wind_obj, stormL_obj):
    storms = {}
    active_storms = []

    gen_dims = genesis_obj._probs.dims
    stoL_dims = [d for d in stormL_obj._pdfs.dims if d not in ['stormL']]
    wind_dims = [d for d in wind_obj._pdfs.dims if d not in ['wind']]

    for iday in range(tmp['weather_0'].shape[0]):
        sst = tmp.sst.iloc[iday]
        weather_dict = {'weather_0':tmp.weather_0.iloc[iday], 'weather_1':tmp.weather_1.iloc[iday], 'weather_2':tmp.weather_2.iloc[iday]}

        ended_storms = []
        for storm_id in active_storms:
            day_id = np.where(np.array(storms[storm_id])>0)[0][-1]
            if day_id+1 == len(storms[storm_id]):
                ended_storms.append(storm_id)
            else:
                wind_before = storms[storm_id][day_id]
                wind_change_before = 0
                if day_id > 1:
                    wind_change_before = storms[storm_id][day_id-1] - storms[storm_id][day_id-2]
                # conditions = {'weather_0':tmp.weather_0.iloc[iday], 'sst':tmp.sst.iloc[iday], 'wind_before':wind_before}
                conditions = {d:tmp[d].iloc[iday] for d in wind_dims if d not in ['wind_before','wind_change_before']}
                for d,v in {'wind_before':wind_before, 'wind_change_before':wind_change_before}.items():
                    if d in wind_dims:
                        conditions[d] = v
                new_wind = wind_obj.sample(conditions)
                storms[storm_id][day_id+1] = max([10,new_wind])
        for sto in ended_storms:
            active_storms.remove(sto)

        # if genesis_obj.sample({'weather_0':tmp.weather_0.iloc[iday], 'weather_1':tmp.weather_1.iloc[iday], 'weather_2':tmp.weather_2.iloc[iday], 'sst':tmp.sst.iloc[iday]}):
        if genesis_obj.sample({d:tmp[d].iloc[iday] for d in gen_dims}):
            storms[iday] = [0] * stormL_obj.sample({d:tmp[d].iloc[iday] for d in stoL_dims})
            storms[iday][0] = 30
            active_storms.append(iday)

    # clean end of season storms
    for storm_id in storms.keys():
        storms[storm_id] = np.array(storms[storm_id])
        storms[storm_id] = storms[storm_id][storms[storm_id]>0]

    return storms



#
