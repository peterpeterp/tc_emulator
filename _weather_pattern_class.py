# basics
import sys, os,pickle, inspect, textwrap, importlib, glob, itertools, inspect, resource
from datetime import datetime, date, timedelta
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
from collections import Counter
import collections

# for plotting
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import matplotlib as mpl
from matplotlib import colors
import cartopy
import cartopy.crs as ccrs

# for shapes
import cartopy.io.shapereader as shapereader
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.ops import cascaded_union

# for analysis
import scipy
from scipy import signal
from scipy.optimize import linear_sum_assignment
from itertools import cycle
from cartopy.util import add_cyclic_point
import math
from scipy import stats


import sklearn
from sklearn import metrics
# from haversine import haversine
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.calibration import calibration_curve
from sklearn import preprocessing
from scipy.stats import ks_2samp
from sklearn import cluster, datasets, mixture

from minisom import MiniSom

def make_symlink(source,target):
	#os.symlink(os.path.abspath(os.getcwd())+'/'+source,target)
	if os.path.isfile(target) == False:
		if sys.platform == 'darwin':
			os.symlink(os.path.abspath(os.getcwd())+'/'+source,target)
		if sys.platform in ['linux','linux2']:
			os.symlink(source,target)

class weather_patterns(object):

	def __init__(self, source, working_directory):
		self._meta = {
					'source':source,
					'variables':[],
					'preprocess':{},
					}
		self._source = source
		self._source_dict = None
		self._data_dir = working_directory

		self._input_data = {}
		self._input_data_details = {}
		self._mapping = {}
		self._clustering = {}

		self._split_tag = ''

	def destroy_tree(self, lvl):
		if 1 > lvl:
			self._dir_lvl1 = None
		if 2 > lvl:
			self._dir_lvl2 = None
		if 3 > lvl:
			self._dir_lvl3 = None
		if 4 > lvl:
			self._dir_lvl4 = None

	def add_data(self, array, variable, cmap='jet'):
		self._meta['variables'] += [variable]
		self._input_data[variable] = array
		self._input_data_details[variable] = {'cmap':cmap}

	def preprocess_harmonize_time(self,charchters=10):
		first,last = [],[]
		for variable,tmp in self._input_data.items():
			self._input_data[variable] = self._input_data[variable].assign_coords(time=np.array([str(tt)[:10] for tt in self._input_data[variable].time.values],'datetime64[ns]'))
			first.append(self._input_data[variable].time[0].values)
			last.append(self._input_data[variable].time[-1].values)

		print(first,last)
		print(np.max(first),np.min(last))
		for variable,tmp in self._input_data.items():
			self._input_data[variable] = self._input_data[variable].loc[np.max(first):np.min(last)]

	def preprocess_select_months(self, months, month_colors=None):
		for variable,tmp in self._input_data.items():
			self._input_data[variable] = tmp[np.isin(tmp.time.dt.month, months), :, :]

		if month_colors is None:
			month_colors = sns.husl_palette(len(months))

		self._months = {
			'mon':months,
			'mon_names':[{1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}[mon] for mon in months],
			'mon_colors': month_colors
		}

		self._meta['preprocess']['months'] = months

	def preprocess_standardized_anomalies(self, ref_period):
		self._meta['preprocess']['standardized_anomalies'] = {'ref_period':ref_period}
		for variable,tmp in self._input_data.items():
			self._input_data[variable].values = (tmp - tmp.loc[ref_period[0]:ref_period[1],:,:].mean(axis=0)) / tmp.loc[ref_period[0]:ref_period[1],:,:].std(axis=0)

	def preprocess_regrid(self, reference_lats, reference_lons):
		self._meta['preprocess']['regrid'] = {'reference_lats':reference_lats, 'reference_lons':reference_lons}
		for variable,tmp in self._input_data.items():
			refrence_grid = xr.Dataset({'time': (['time'], tmp.time.data),'lat': (['lat'], reference_lats),'lon': (['lon'], reference_lons),})
			regridder = xe.Regridder(tmp, refrence_grid, 'bilinear')
			self._input_data[variable] = regridder(tmp)

	#######################
	# lvl0 : input vector
	#######################

	def preprocess_create_vector(self, variables):
		first = True
		for variable in variables:
			tmp = self._input_data[variable]
			if first:
				vector = tmp.values.reshape((tmp.shape[0],-1))
				xx = [variable] * vector.shape[1]
				first = False
			else:
				vector = np.hstack((vector, tmp.values.reshape((tmp.shape[0],-1))))
				xx += [variable] * tmp.values.reshape((tmp.shape[0],-1)).shape[1]

		vector[np.isnan(vector)] = 0
		tt = np.array([self._source + '_' + str(tt)[:10] for tt in tmp.time.values])
		self._vector = xr.DataArray(vector, coords=dict(tt=tt, xx=range(vector.shape[1])), dims=['tt','xx'])
		self._vector_time = xr.DataArray(tmp.time, coords=dict(tt=self._vector.tt), dims=['tt'])
		self._vector_source = xr.DataArray(np.array([self._source]*len(tt)), coords=dict(tt=self._vector.tt), dims=['tt'])
		self._vector_var = xr.DataArray(np.array(xx), coords=dict(xx=range(vector.shape[1])), dims=['xx'])
		self._lon = self._input_data[variable].lon
		self._lat = self._input_data[variable].lat

		self._decades = {'dec':np.unique([int(int(yr/10)*10) for yr in self._vector_time.dt.year.values])}
		self._decades['dec_names'] = [str(dec) for dec in self._decades['dec']]
		self._decades['dec_colors'] = sns.husl_palette(len(self._decades['dec']))

	def store_input(self, tag):
		self._dir_lvl0 = self._data_dir+'/'+tag

		if os.path.isdir(self._dir_lvl0) == False: os.system('mkdir -p '+self._dir_lvl0)
		out = {'meta':self._meta, 'input_data_details':self._input_data_details, 'months':self._months, 'decades':self._decades, 'lon':self._lon, 'lat':self._lat}
		with open(self._dir_lvl0+'/input.pkl', 'wb') as outfile:
			pickle.dump(out, outfile)

		log = open(self._dir_lvl0+'/meta.txt','w')
		log.write(str(self._meta))
		log.close()

		if os.path.isfile(self._dir_lvl0+'/vector.nc'):
			os.system('rm '+self._dir_lvl0+'/vector.nc')
		xr.Dataset({'vector':self._vector, 'vector_time':self._vector_time, 'vector_source':self._vector_source, 'vector_var':self._vector_var}).to_netcdf(self._dir_lvl0+'/vector.nc')

	def load_input(self, tag, years=None):
		self.destroy_tree(0)
		self._dir_lvl0 = self._data_dir+'/'+tag
		with open(self._dir_lvl0+'/input.pkl', 'rb') as infile:
			tmp = pickle.load(infile)
		self._input_data_details = tmp['input_data_details']
		self._meta = tmp['meta']
		self._months = tmp['months']
		self._decades = tmp['decades']
		self._lon = tmp['lon']
		self._lat = tmp['lat']

		nc_vector = xr.load_dataset(self._dir_lvl0+'/vector.nc')
		self._vector = nc_vector['vector']
		self._vector_time = nc_vector['vector_time']
		self._vector_source = nc_vector['vector_source']
		self._vector_var = nc_vector['vector_var']

		if years is not None:
			valid = np.isin(nc_vector['vector_time'].dt.year.values, years)
			self._vector = self._vector[valid,:]
			self._vector_time = self._vector_time[valid]
			self._vector_source = self._vector_source[valid]

	def merge_vectors(self, paths):
		self._dir_lvl0 = self._data_dir+'/'+tag
		if os.path.isdir(self._dir_lvl0) == False: os.system('mkdir -p '+self._dir_lvl0)

		vector_files = [p + '/vector.nc' for p in paths if os.path.isfile(p + '/vector.nc')]
		xr.open_mfdataset(vector_files, concat_dim='tt', combine='by_coords').to_netcdf(self._dir_lvl0+'/vector.nc')

	#######################
	# lvl1 : dimensionality reduction
	#######################

	def set_split(self, years, split_tag=None):
		if split_tag is None:
			self._split_tag = '_%s-%s' % (min(years),max(years))
		else:
			self._split_tag = split_tag
		self._vector = self._vector[np.isin(self._vector_time.dt.year,years)]
		self._vector_source = self._vector_source[np.isin(self._vector_time.dt.year,years)]
		self._vector_time = self._vector_time[np.isin(self._vector_time.dt.year,years)]

	def mapping_raw(self):
		self._mapping_tag = 'mapping_raw'
		self._dir_lvl1 = self._dir_lvl0+'/'+self._mapping_tag+self._split_tag
		if os.path.isdir(self._dir_lvl1) == False:
			os.mkdir(self._dir_lvl1)
		xr.Dataset({'vector':self._vector, 'vector_time':self._vector_time, 'vector_source':self._vector_source, 'vector_var':self._vector_var}).to_netcdf(self._dir_lvl1+'/mapped_vector.nc')
		# if os.path.isfile(self._dir_lvl1+'/mapped_vector.nc'):
		# 	os.system('rm '+self._dir_lvl1+'/mapped_vector.nc')
		# make_symlink(self._dir_lvl0 +'/vector.nc',self._dir_lvl1+'/mapped_vector.nc')

	def mapping_wrapper(self, mapping_function, tag, overwrite=False):
		self.destroy_tree(1)
		self._dir_lvl1 = self._dir_lvl0+'/'+tag+self._split_tag
		self._pre_mapping_tag = tag
		if os.path.isdir(self._dir_lvl1) == False: os.mkdir(self._dir_lvl1)
		log = open(self._dir_lvl1+'/mapping_function.py','w')
		log.write(inspect.getsource(mapping_function))
		log.close()
		file_name = self._dir_lvl1+'/mapped_vector.nc'
		if os.path.isfile(file_name) and overwrite==False:
			self.load_mapping(tag)
		else:
			proj,add_results = mapping_function(self._vector.values)
			self._pre_mapping = xr.DataArray(proj, coords={'tt':self._vector.tt, 'xx':range(proj.shape[1])}, dims=['tt','xx'])
			xr.Dataset({'vector':self._pre_mapping}).to_netcdf(file_name)

			if add_results is not None:
				with open(file_name.replace('mapped_vector.nc','additional_results_from_mapping.pkl'), 'wb') as outfile:
					pickle.dump(add_results, outfile)

	def load_mapping(self, tag):
		self.destroy_tree(1)
		self._dir_lvl1 = self._dir_lvl0+'/'+tag+self._split_tag
		self._pre_mapping_tag = tag
		file_name = self._dir_lvl1+'/mapped_vector.nc'
		if os.path.isfile(file_name):
			self._pre_mapping = xr.load_dataset(file_name)['vector']
			return 1
		else:
			self._pre_mapping = None
			print('mapping not found: ',file_name)
			return 0

	def plot_mapping(self, name_addon=''):
		plt.close('all')
		with PdfPages(self._dir_lvl1+'/mapping'+name_addon+'.pdf') as pdf:
			# fig, ax = plt.subplots(nrows=1)
			# ax.scatter(self._pre_mapping[:,0],self._pre_mapping[:,1], s=5, color='b')
			# pdf.savefig(bbox_inches='tight'); plt.close()

			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
			for month,color,month_name in zip(self._months['mon'], self._months['mon_colors'], self._months['mon_names']):
				if self._source_dict is not None:
					for source,marker in self._source_dict.items():
						ids = (self._vector_time.dt.month.values == month) & (self._vector_source.values == source)
						ax.scatter(self._pre_mapping[ids,0],self._pre_mapping[ids,1], s=5, color=color, alpha=0.3, marker=marker, label=month_name+' '+source)
				else:
					ids = self._vector_time.dt.month.values == month
					ax.scatter(self._pre_mapping[ids,0],self._pre_mapping[ids,1], s=5, color=color, alpha=0.3, label=month_name)

			box = ax.get_position()
			ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
			pdf.savefig(bbox_inches='tight'); plt.close()

	#######################
	# lvl2 : clustering
	#######################

	def clustering_wrapper(self, clustering_function, tag, overwrite=False):
		self.destroy_tree(2)
		self._dir_lvl2 = self._dir_lvl1+'/'+tag
		self._clustering_tag = tag
		if overwrite:
			os.system('rm -r '+self._dir_lvl2)
		if os.path.isdir(self._dir_lvl2) == False:
			os.mkdir(self._dir_lvl2)
		log = open(self._dir_lvl2+'/clustering_function.py','w')
		log.write(inspect.getsource(clustering_function))
		log.close()
		file_name = self._dir_lvl2+'/centers.nc'
		if os.path.isfile(file_name):
			self.load_cluster_centers_and_labels(tag)
		else:
			# centers and labels in the mapped space
			results = clustering_function(self._pre_mapping.values)
			if 'other_results' in results.keys():
				with open(file_name.replace('centers.nc','additional_results_from_clustering.pkl'), 'wb') as outfile:
					pickle.dump(results['other_results'], outfile)

			centers_map = results['centers_map']
			labels_map = pairwise_distances(np.array(centers_map),self._pre_mapping[:,:].values).argmin(axis=0)

			# centers and labels in real space
			centers = []
			for lab in np.sort(np.unique(labels_map)):
				centers.append(self._vector[labels_map==lab,:].mean(axis=0).values)
			labels = pairwise_distances(np.array(centers),self._vector[:,:]).argmin(axis=0)

			self._clust_labels = xr.DataArray(labels, coords={'tt':self._vector.tt}, dims=['tt'])
			xr.Dataset({'labels':self._clust_labels}).to_netcdf(file_name.replace('centers.nc','labels.nc'))
			self._clust_centers = xr.DataArray(centers, coords={'label':np.sort(np.unique(labels)),'xx':self._vector.xx}, dims=['label','xx'])
			xr.Dataset({'centers':self._clust_centers}).to_netcdf(file_name)

	def add_external_cluster_centers(self, tag, external_dir_lvl2, post_mapping_external=None):
		self.destroy_tree(2)
		self._dir_lvl2 = self._dir_lvl1+'/'+tag
		self._clustering_tag = tag
		if os.path.isdir(self._dir_lvl2):
			os.system('rm -r '+self._dir_lvl2)
		os.mkdir(self._dir_lvl2)
		xr.load_dataset(external_dir_lvl2+'/centers.nc').to_netcdf(self._dir_lvl2+'/centers.nc')
		if os.path.isfile(external_dir_lvl2+'/additional_results_from_clustering.pkl'):
			os.system('cp '+external_dir_lvl2+'/additional_results_from_clustering.pkl '+self._dir_lvl2+'/additional_results_from_clustering.pkl')
		if post_mapping_external is not None:
			os.mkdir(self._dir_lvl2+'/'+post_mapping_external+'_ext')
			os.system('cp '+external_dir_lvl2+'/'+post_mapping_external+'/mapped_centers.nc '+self._dir_lvl2+'/'+post_mapping_external+'_ext/mapped_centers.nc')

		self._clust_centers = xr.open_dataset(self._dir_lvl2+'/centers.nc')['centers']

	def add_external_cluster_labels(self, tag, external_dir_lvl2=None):
		self.destroy_tree(2)
		self._dir_lvl2 = self._dir_lvl1+'/'+tag
		self._clustering_tag = tag
		if os.path.isdir(self._dir_lvl2):
			os.system('rm -r '+self._dir_lvl2)
		os.system('mkdir -p '+self._dir_lvl2)
		make_symlink(external_dir_lvl2+'/labels.nc',self._dir_lvl2+'/labels.nc')
		make_symlink(external_dir_lvl2+'/additional_results_from_clustering.pkl',self._dir_lvl2+'/additional_results_from_clustering.pkl')
		self._clust_labels = xr.open_dataset(self._dir_lvl2+'/labels.nc')['labels']

		# centers and labels in real space
		centers = []
		for lab in np.sort(np.unique(self._clust_labels)):
			centers.append(self._vector[self._clust_labels==lab,:].mean(axis=0).values)

		self._clust_centers = xr.DataArray(centers, coords={'label':np.sort(np.unique(self._clust_labels)),'xx':self._vector.xx}, dims=['label','xx'])
		xr.Dataset({'centers':self._clust_centers}).to_netcdf(self._dir_lvl2+'/centers.nc')

	def assign_data_to_clusters(self):
		file_name = self._dir_lvl2+'/labels.nc'

		# if os.path.isfile(file_name):
		# 	self._clust_labels = xr.open_dataset(self._dir_lvl2+'/labels.nc')['labels']
		# else:
		self._clust_labels = xr.DataArray(pairwise_distances(np.array(self._clust_centers),self._vector[:,:]).argmin(axis=0), coords={'tt':self._vector.tt}, dims=['tt'])
		xr.Dataset({'labels':self._clust_labels}).to_netcdf(file_name)

	def assign_data_to_clusters_winner(self):
		file_name = self._dir_lvl2+'/labels.nc'

		if os.path.isfile(self._dir_lvl2+'/additional_results_from_clustering.pkl'):
			with open(self._dir_lvl2+'/additional_results_from_clustering.pkl', 'rb') as infile:
				som = pickle.load(infile)['SOM']

				labels = np.array([pos[0]*self._nrows + pos[1] for pos in [som.winner(self._vector[i,:].values) for i in range(self._vector.shape[0])]])

				self._clust_labels = xr.DataArray(labels, coords={'tt':self._vector.tt}, dims=['tt'])
				xr.Dataset({'labels':self._clust_labels}).to_netcdf(file_name)

				print(np.unique(self._clust_labels))

		else:
			print('no SOM found')

	def load_cluster_centers_and_labels(self, tag):
		self.destroy_tree(2)
		self._dir_lvl2 = self._dir_lvl1+'/'+tag
		self._clustering_tag = tag
		self._clust_labels = xr.open_dataset(self._dir_lvl2+'/labels.nc')['labels']
		self._clust_centers = xr.open_dataset(self._dir_lvl2+'/centers.nc')['centers']
		if os.path.isfile(self._dir_lvl2+'/additional_results_from_clustering.pkl'):
			with open(self._dir_lvl2+'/additional_results_from_clustering.pkl', 'rb') as infile:
				self._clust_add_info = pickle.load(infile)
		else:
			self._clust_add_info = {}
		print(self._clust_add_info)

	def merge_labels_from_different_folders(self, label_files):
		if hasattr(self, '_dir_lvl2'):
			if os.path.isfile(self._dir_lvl2+'/labels.nc') == False:
				xr.open_mfdataset(label_files, concat_dim='tt', combine='by_coords').to_netcdf(self._dir_lvl2+'/labels.nc')
		else:
			print('execute "add_external_cluster_labels()" first')

	########################
	# lvl3 : post mapping
	########################

	def load_mapping_for_plotting(self, tag):
		print( self._dir_lvl2, tag)
		self.destroy_tree(3)
		self._post_mapping_tag = tag
		print( self._dir_lvl2, tag)

		self._dir_lvl3 = self._dir_lvl2+'/'+tag
		if os.path.isdir(self._dir_lvl3) == False:
			os.mkdir(self._dir_lvl3)

		file_name = self._dir_lvl0+'/'+tag+'/mapped_vector.nc'
		if os.path.isfile(file_name):
			self._mapping = xr.load_dataset(file_name)['vector']
			return 1
		else:
			self._mapping = None
			print('mapping not found: ',file_name)
			return 0

	def map_cluster_centers(self, overwrite=False):
		file_name = self._dir_lvl3+'/mapped_centers.nc'

		if os.path.isfile(file_name) and overwrite==False:
			self._clust_centers_map = xr.open_dataset(file_name)['centers_map']
		else:
			centers_map = []
			for lab in np.unique(self._clust_labels):
				centers_map.append(self._mapping[self._clust_labels==lab,:].mean(axis=0).values)
			centers_map = np.array(centers_map)

			self._clust_centers_map = xr.DataArray(centers_map, coords={'label':np.unique(self._clust_labels),'xx':range(centers_map.shape[1])}, dims=['label','xx'])

			xr.Dataset({'centers_map':self._clust_centers_map}).to_netcdf(file_name)

	########################
	# lvl4 : grid
	########################

	def clusters_assign_to_grid(self, nrows, ncols):
		self._dir_lvl4 = self._dir_lvl3+'/grid_%sx%s' % (nrows,ncols)
		if os.path.isdir(self._dir_lvl4) == False:	os.mkdir(self._dir_lvl4); os.mkdir(self._dir_lvl4+'/stats')
		file_name = self._dir_lvl4+'/grid_%sx%s.pkl' % (nrows,ncols)
		self._nrows, self._ncols = nrows,ncols

		if os.path.isfile(file_name):
			with open(file_name, 'rb') as infile:
				tmp = pickle.load(infile)
			self._grid_labels = tmp['grid_labels']
			self._axes_grid = tmp['axes_grid']
		else:
			centers_map = self._clust_centers_map
			xmin,xmax,ymin,ymax = centers_map[:,0].min().values,centers_map[:,0].max().values,centers_map[:,1].min().values,centers_map[:,1].max().values
			grid = np.meshgrid(np.linspace(ymin,ymax,nrows),np.linspace(xmin,xmax,ncols))
			grid_points = []
			for ix,x in enumerate(grid[1][:,0]):
				for iy,y in enumerate(grid[0][0,:]):
					grid_points.append([x,y])
			grid_points = np.array(grid_points)

			if 'SOM' in self._clust_add_info.keys():
				som_label_pos = self._clust_centers_map.label.values.reshape((self._nrows,self._ncols))
				self._grid_labels = som_label_pos.flatten()
				self._axes_grid = []
				for lab in self._grid_labels:
					tmp = np.where(som_label_pos==lab)
					self._axes_grid.append([tmp[0][0],tmp[1][0]])

			else:
				centers_map = self._clust_centers_map
				xmin,xmax,ymin,ymax = centers_map[:,0].min().values,centers_map[:,0].max().values,centers_map[:,1].min().values,centers_map[:,1].max().values
				grid = np.meshgrid(np.linspace(ymin,ymax,nrows),np.linspace(xmin,xmax,ncols))
				axes_grid = []
				for ix,x in enumerate(grid[1][:,0]):
					for iy,y in enumerate(grid[0][0,:]):
						axes_grid.append([nrows-iy-1,ix])
				self._axes_grid = np.array(axes_grid)

				cost = pairwise_distances(np.array(grid_points),np.array(centers_map))
				row_ind, self._grid_labels = linear_sum_assignment(cost)

			self._axes_grid = np.array(self._axes_grid)

			plt.close('all')
			with PdfPages(self._dir_lvl4+'/clustering_centers_%sx%s.pdf' % (nrows,ncols)) as pdf:
				for lab in np.unique(self._clust_labels):
					if self._mapping is not None:
						plt.scatter(self._mapping[self._clust_labels==lab,0],self._mapping[self._clust_labels==lab,1], s=5, color=self._lab_colors[lab-1], alpha=0.3)
					plt.annotate(lab, xy=(grid_points[self._grid_labels==lab,0],grid_points[self._grid_labels==lab,1]), color=self._lab_colors[lab-1], backgroundcolor='k')
					plt.scatter(grid_points[self._grid_labels==lab,0],grid_points[self._grid_labels==lab,1], s=1)
				plt.axis('off')
				plt.title('%s on %s mapping' %(self._clustering_tag,self._post_mapping_tag))
				pdf.savefig(bbox_inches='tight'); plt.close()

			out = {'grid_labels':self._grid_labels, 'axes_grid':self._axes_grid}
			with open(file_name, 'wb') as outfile:
				pickle.dump(out, outfile)

	########################
	# stats
	########################

	def stats_TC(self, file=None, max_lag=7, overwrite=False):
		out_file = self._dir_lvl4+'/stats/stats_TC.csv'
		if os.path.isfile(out_file) and overwrite==False:
			self._tracks = pd.read_csv(out_file)
		else:
			self._tracks = pd.read_csv(file)
			self._tracks.time = np.array(self._tracks.time, np.datetime64)

			self._tracks = self._tracks.loc[np.isin(self._tracks.month,self._months['mon'])]

			# polygon=[[self._lon.min(),self._lat.max()],[self._lon.min(),self._lat.min()],[self._lon.max(),self._lat.min()],[self._lon.max(),self._lat.max()]]
			# locs = np.array([[x,y] for x,y in zip(self._tracks['lon'].values,self._tracks['lat'].values)])
			# relevant_location = matplotlib.path.Path(polygon).contains_points(locs)

			for lag in range(max_lag):
				self._tracks['label_lag'+str(int(lag))] = np.nan
				for lab in np.unique(self._clust_labels):
					group = np.where(self._clust_labels==lab)[0]

					group_cut = group[group+lag<self._vector_time.shape[0]]
					tmp_group = []
					for year in np.unique(self._vector_time.dt.year):
						tmp_group += list(group_cut[(self._vector_time[group_cut].dt.year.values == year) & (self._vector_time[group_cut+lag].dt.year.values == year)] + lag)

					relevant_times = np.isin(self._tracks.time.values, self._vector_time[tmp_group].values)

					relevant = relevant_times# & relevant_location

					self._tracks.loc[relevant,'label_lag'+str(int(lag))] = lab

			self._tracks.to_csv(out_file)

	def event_density(self, lag=0, overwrite=False):
		file_name = self._dir_lvl4+'/stats/stats_event_density.nc'
		if overwrite:
			os.system('rm '+file_name)
		if os.path.isfile(file_name):
			self._event_density = xr.open_dataset(file_name)['density']
			if lag in self._event_density.lag.values:
				return 0

		density = xr.DataArray(np.zeros([len(np.unique(self._clust_labels)),1,len(self._lat),len(self._lon)]), coords={'label':np.unique(self._clust_labels),'lag':[lag],'lat':self._lat,'lon':self._lon}, dims=['label','lag','lat','lon'])
		dx,dy = 0.5*np.diff(self._lon,1)[0], 0.5*np.diff(self._lat,1)[0]

		for lab in np.unique(self._clust_labels):
			relevant_ids = (self._tracks['label_lag'+str(int(lag))] == lab)
			if np.sum(relevant_ids) > 0:
				points = np.array([self._tracks.loc[relevant_ids,'lon'].values,self._tracks.loc[relevant_ids,'lat'].values]).T

				for iy,y in enumerate(self._lat):
					for ix,x in enumerate(self._lon):
						density.loc[lab,lag,y,x] = matplotlib.path.Path([(x-dx,y-dy),(x+dx,y-dy),(x+dx,y+dy),(x-dx,y+dy)]).contains_points(points).sum() / float(np.sum(self._clust_labels==lab))

		if os.path.isfile(file_name) == False:
			self._event_density = density
		else:
			self._event_density = xr.concat((self._event_density,density), dim='lag')

		xr.Dataset({'density':self._event_density}).to_netcdf(file_name)

	def stats_frequency(self):#
		file_name = self._dir_lvl4+'/stats/stats_***.nc'
		if len(glob.glob(file_name)) == 6:
			self._stats = {}
			for name in ['count', 'freq', 'freq_mon', 'freq_dec', 'freq_year', 'freq_trend']:
				self._stats[name] = xr.open_dataset(file_name.replace('***',name))[name]
		else:
			self._stats = {}
			self._stats['count'] = xr.DataArray(np.zeros([len(np.unique(self._clust_labels))])*np.nan, coords={'labels':np.unique(self._clust_labels)}, dims=['labels'])

			self._stats['freq'] = self._stats['count'].copy()
			for lab in np.unique(self._clust_labels):
				self._stats['count'].loc[lab] = len(np.where(self._clust_labels == lab)[0])
				self._stats['freq'].loc[lab] = len(np.where(self._clust_labels == lab)[0]) / float(len(self._clust_labels))

			self._stats['freq_mon'] = xr.DataArray(np.zeros([len(np.unique(self._clust_labels)),len(self._months['mon'])])*np.nan, coords={'labels':np.unique(self._clust_labels), 'months':self._months['mon_names']}, dims=['labels','months'])
			for lab in np.unique(self._clust_labels):
				for month,mon_name in zip(self._months['mon'],self._months['mon_names']):
					self._stats['freq_mon'].loc[lab,mon_name] = len(np.where((self._clust_labels == lab) & (self._vector_time.dt.month == month))[0])/float(len(np.where(self._clust_labels == lab)[0]))

			self._stats['freq_dec'] = xr.DataArray(np.zeros([len(np.unique(self._clust_labels)),len(self._decades['dec'])])*np.nan, coords={'labels':np.unique(self._clust_labels), 'decades':self._decades['dec_names']}, dims=['labels','decades'])
			for lab in np.unique(self._clust_labels):
				for dec,dec_name in zip(self._decades['dec'],self._decades['dec_names']):
					self._stats['freq_dec'].loc[lab,dec_name] = len(np.where((self._clust_labels == lab) & (self._vector_time.dt.year >= dec) & (self._vector_time.dt.year < dec+10))[0])/float(len(np.where(self._clust_labels == lab)[0]))

			years = np.unique(self._vector_time.dt.year)
			self._stats['freq_year'] = xr.DataArray(np.zeros([len(np.unique(self._clust_labels)),len(years)])*np.nan, coords={'labels':np.unique(self._clust_labels), 'year':years}, dims=['labels','year'])
			self._stats['freq_trend'] = xr.DataArray(np.zeros([len(np.unique(self._clust_labels)),3])*np.nan, coords={'labels':np.unique(self._clust_labels), 'result':['slope', 'intercept', 'p_value']}, dims=['labels','result'])
			for lab in np.unique(self._clust_labels):
				occ = self._clust_labels.copy()
				occ.values = np.array(occ==lab, np.int)
				occ = occ.assign_coords(tt=self._vector_time)
				occ = occ.groupby('tt.year').mean('tt')
				self._stats['freq_year'].loc[lab] = occ.values
				slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(occ.year.values,occ.values)
				self._stats['freq_trend'].loc[lab,:] = [slope, intercept, p_value]

			for name,item in self._stats.items():
				xr.Dataset({name:item}).to_netcdf(file_name.replace('***',name))

	def merge_stats_from_different_folders(self, stat_folders):
		if hasattr(self, '_dir_lvl4'):

			tags = ['_'.join(fl.split('/')[-6].split('_')[:-1]) for fl in stat_folders]
			weights = xr.DataArray( coords={'ver':tags}, dims=['ver'])
			weights.values = [np.isin(['_'.join(tt.split('_')[:-1]) for tt in self._clust_labels.tt.values],tag).sum() / float(self._clust_labels.shape[0]) for tag in tags]

			self._stats = {}
			for name in ['count', 'freq', 'freq_mon', 'freq_dec', 'freq_year', 'freq_trend']:
				files = [fl + '/stats_'+name+'.nc' for fl in stat_folders]
				tmp = xr.open_mfdataset(files, concat_dim='ver', combine='nested')[name]
				tmp = tmp.assign_coords(ver=tags) * weights
				self._stats[name] = tmp.sum('ver')

			files = [fl + '/stats_TC.csv' for fl in stat_folders]
			li = []
			if len(files) == len(tags):
				for filename in files:
					df = pd.read_csv(filename)
					li.append(df)
				self._tracks = pd.concat(li, axis=0, ignore_index=True)

		else:
			print('execute "define_plot_environment()" first')


	#######################
	# clustering figures
	#######################

	def define_plot_environment(self, pre_mapping, clustering, post_mapping, nrows, ncols, label_colors=None):
		self.load_mapping(pre_mapping)
		self.load_cluster_centers_and_labels(clustering)
		self.load_mapping_for_plotting(post_mapping)
		self.map_cluster_centers()

		if label_colors is None:
			label_colors = sns.husl_palette(self._clust_centers.label.shape[0])
		self._lab_colors = label_colors

		self.clusters_assign_to_grid(nrows, ncols)
		print(self._dir_lvl4)

	def plot_labels(self):
		labels = self._clust_labels
		mapping = self._mapping

		plt.close('all')
		with PdfPages(self._dir_lvl4+'/scatter_labels.pdf') as pdf:
			# for lab in np.unique(labels):
			# 	plt.scatter(mapping[labels_map==lab,0],mapping[labels_map==lab,1], s=5, color=self._lab_colors[lab])
			# 	plt.scatter(mapping[labels_map==lab,0].mean(),mapping[labels_map==lab,1].mean(), s=10, marker='*', c='k')
			# 	plt.annotate(lab, xy=(mapping[labels_map==lab,0].mean(),mapping[labels_map==lab,1].mean()), c='k')
			#
			# plt.title('%s on %s mapping' %(self._clustering_tag,self._mapping_tag))
			# pdf.savefig(bbox_inches='tight'); plt.close()

			for lab in np.unique(labels):
				plt.scatter(mapping[labels==lab,0],mapping[labels==lab,1], s=5, color=self._lab_colors[lab])
				plt.scatter(mapping[labels==lab,0].mean(),mapping[labels==lab,1].mean(), s=10, marker='*', c='k')
				plt.annotate(lab, xy=(mapping[labels==lab,0].mean(),mapping[labels==lab,1].mean()), c='k')

			plt.title('%s on %s mapping - real clusters' %(self._clustering_tag,self._post_mapping_tag))
			pdf.savefig(bbox_inches='tight'); plt.close()

	def plot_fields(self, name_addon='', style='contourf'):
		plt.close('all')
		for field_name in np.unique(self._vector_var):
			fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._nrows*3,self._ncols*4*(len(self._lat)/len(self._lon))), subplot_kw={'projection': ccrs.PlateCarree()})
			tmp_field = self._vector[:,(self._vector_var==field_name)].values.reshape(self._vector.shape[0],len(self._lat),len(self._lon))
			for lab in np.unique(self._clust_labels):
				ax = axes[self._axes_grid[self._grid_labels==lab][0][0],self._axes_grid[self._grid_labels==lab,1][0]]
				ax.annotate(lab, xy=(0,0), xycoords='axes fraction', backgroundcolor='w')
				ax.coastlines()
				ax.set_extent([self._lon.min(),self._lon.max(),self._lat.min(),self._lat.max()], crs=ccrs.PlateCarree())
				z = tmp_field[self._clust_labels.values==lab,:,:].mean(0)
				if style=='contourf':
					im = ax.contourf(self._lon,self._lat,z, vmin=-1.5, vmax=1.5, cmap=self._input_data_details[field_name]['cmap'])
				if style=='pcolormesh':
					im = ax.pcolormesh(self._lon,self._lat,z, vmin=-1.5, vmax=1.5, cmap=self._input_data_details[field_name]['cmap'])


			im = plt.scatter(np.linspace(-1.5,1.5,10),np.linspace(-1.5,1.5,10), c=np.linspace(-1.5,1.5,10), cmap=self._input_data_details[field_name]['cmap'])

			fig.colorbar(im, ax=axes[:, :], location='right', shrink=0.6,label=field_name+' [sigma]')
			plt.savefig(self._dir_lvl4+'/fields_'+field_name+name_addon+'.png', transparent=True, dpi=300, bbox_inches='tight'); plt.close()

	def plot_events(self, lag=0, distance=2, indicator='ACE', highlight='genesis', additional_area=0, legend_values=[], legend_name=''):
		plt.close('all')
		fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._ncols*3,self._nrows*3*(len(self._lat)/len(self._lon))), subplot_kw={'projection': ccrs.PlateCarree()})

		for lab in np.unique(self._clust_labels):
			ax = axes[self._axes_grid[self._grid_labels==lab][0][0],self._axes_grid[self._grid_labels==lab,1][0]]
			ax.annotate(lab, xy=(1,1), xycoords='axes fraction', backgroundcolor='w', ha='center', va='center')
			ax.coastlines()
			ax.set_extent([self._lon.min()-additional_area,self._lon.max()+additional_area,self._lat.min()-additional_area,self._lat.max()+additional_area], crs=ccrs.PlateCarree())

			relevant_ids = np.where(self._tracks['label_lag'+str(lag)]==lab)[0]

			marker_factor = 50 / self._tracks[indicator].max()
			if len(relevant_ids) > 0:
				p1 = ax.scatter(self._tracks.lon.values[relevant_ids], self._tracks.lat.values[relevant_ids], s=self._tracks[indicator].values[relevant_ids]*marker_factor, marker='.', color='gray', alpha=0.5, label='all storms')

				d_relevant = np.where(self._tracks.distance.values[relevant_ids] <= distance)[0]
				if len(d_relevant) > 0:
					p2 = ax.scatter(self._tracks.lon.values[relevant_ids[d_relevant]], self._tracks.lat.values[relevant_ids[d_relevant]], marker='.', color='m', s=self._tracks[indicator].values[relevant_ids[d_relevant]]*marker_factor, alpha=0.5, label='storms closer than\n2° from and island')

				if highlight == 'genesis':
					h_relevant = np.where(self._tracks[highlight].values[relevant_ids] > 0)[0]
					if len(h_relevant) > 0:
						p3 = ax.scatter(self._tracks.lon.values[relevant_ids[h_relevant]], self._tracks.lat.values[relevant_ids[h_relevant]], s=20, marker='*', color='orange', alpha=0.5, label=highlight)

		for s in legend_values:
			plt.scatter([-99],[-99],s=s*marker_factor, marker='.', color='gray', alpha=0.5, label='%s %s' %(s, legend_name))
		plt.legend(bbox_to_anchor=(1.05, 1), loc='lower left')
		plt.savefig(self._dir_lvl4+'/events_lag'+str(lag)+'_'+indicator+'.png', dpi=300, bbox_inches='tight'); plt.close()

	def plot_genesis_density(self, lats=np.arange(5,35,5), lons=np.arange(-110,-10,10), **kwargs):

		if 'vmin' not in kwargs.keys():		kwargs['vmin'] = 0
		if 'vmax' not in kwargs.keys():		kwargs['vmax'] = 0.01
		plt.close('all')

		fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._nrows*3,self._ncols*4*(len(self._lat)/len(self._lon))), subplot_kw={'projection': ccrs.PlateCarree()})

		for lab in np.unique(self._clust_labels):
			ax = axes[self._axes_grid[self._grid_labels==lab][0][0],self._axes_grid[self._grid_labels==lab,1][0]]
			ax.annotate(lab, xy=(0,0), xycoords='axes fraction', backgroundcolor='w')
			ax.coastlines()

			tmp = self._tracks.loc[(self._tracks.label_lag0 == lab) & (self._tracks.genesis)]
			kernel = stats.gaussian_kde(tmp[['lon','lat']].values.T)
			yy,xx = np.meshgrid(lats,lons)
			Z = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)

			im = ax.contourf(xx,yy, Z, **kwargs)
		plt.savefig(self._dir_lvl4+'/genesis_density.png', dpi=300, bbox_inches='tight'); plt.close()

		im = plt.scatter(np.linspace(kwargs['vmin'],kwargs['vmax'],10),np.linspace(kwargs['vmin'],kwargs['vmax'],10), c=np.linspace(kwargs['vmin'],kwargs['vmax'],10), **kwargs)
		fig,ax = plt.subplots(nrows=1, figsize=(2,0.5))
		cb = fig.colorbar(im, cax=ax, orientation='horizontal',label='frequency of events')
		plt.savefig(self._dir_lvl4+'/genesis_density_cmap.png', bbox_inches='tight',dpi=300); plt.close()

	def plot_event_density(self, lag=0, style='pcolormesh', **kwargs):

		if 'vmin' not in kwargs.keys():		kwargs['vmin'] = 0
		if 'vmax' not in kwargs.keys():		kwargs['vmax'] = 0.01

		plt.close('all')
		fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._nrows*3,self._ncols*4*(len(self._lat)/len(self._lon))), subplot_kw={'projection': ccrs.PlateCarree()})

		for lab in np.unique(self._clust_labels):
			ax = axes[self._axes_grid[self._grid_labels==lab][0][0],self._axes_grid[self._grid_labels==lab,1][0]]
			ax.annotate(lab, xy=(0,0), xycoords='axes fraction', backgroundcolor='w')
			ax.coastlines()
			ax.set_extent([self._lon.min(),self._lon.max(),self._lat.min(),self._lat.max()], crs=ccrs.PlateCarree())

			if style=='contourf':
				im = ax.contourf(self._lon,self._lat, self._event_density.loc[lab,lag], **kwargs)
			if style=='pcolormesh':
				im = ax.pcolormesh(self._lon,self._lat, self._event_density.loc[lab,lag], **kwargs)

		plt.savefig(self._dir_lvl4+'/event_density_lag'+str(lag)+'.png', bbox_inches='tight',dpi=300); plt.close()

		im = plt.scatter(np.linspace(kwargs['vmin'],kwargs['vmax'],10),np.linspace(kwargs['vmin'],kwargs['vmax'],10), c=np.linspace(kwargs['vmin'],kwargs['vmax'],10), **kwargs)
		fig,ax = plt.subplots(nrows=1, figsize=(2,0.5))
		cb = fig.colorbar(im, cax=ax, orientation='horizontal',label='frequency of events')
		plt.savefig(self._dir_lvl4+'/event_density_lag'+str(lag)+'_cmap.png', bbox_inches='tight',dpi=300); plt.close()

	def plot_stats(self,indicator='ACE',distance=2, max_lag=7):
		plt.close('all')
		for relative_to_storms,name_addon in zip([False,True],['abs','rel']):
			fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._nrows*2,self._ncols*2), sharex=True, sharey=True)

			for lab in np.unique(self._clust_labels):
				ax = axes[self._axes_grid[self._grid_labels==lab,0][0],self._axes_grid[self._grid_labels==lab,1][0]]
				ax.annotate(lab, xy=(0,0), xycoords='axes fraction')

				for lag in range(max_lag):
					if relative_to_storms:
						all_count = np.sum(self._tracks['label_lag'+str(lag)]==lab)
					else:
						all_count = np.sum(self._clust_labels==lab)
					alpha = np.max([np.min([np.sum(self._tracks['label_lag'+str(lag)]==lab) / np.sum(self._clust_labels==lab),1]),0])
					y = self._tracks.loc[(self._tracks['label_lag'+str(lag)]==lab), indicator].sum() / all_count
					ax.bar(x=lag, height=y, width=0.7, color='b', alpha=alpha)
					y = self._tracks.loc[(self._tracks['label_lag'+str(lag)]==lab) & (self._tracks.distance < distance), indicator].sum() / all_count
					ax.bar(x=lag, height=y, width=0.7, color='m', alpha=alpha)

				if self._axes_grid[self._grid_labels==lab,0][0] == self._nrows-1:
					ax.set_xlabel('lag [days]')
				else:
					ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

				if self._axes_grid[self._grid_labels==lab,1][0] == 0:
					ax.set_ylabel(indicator)
				else:
					ax.tick_params(axis='y',which='both',right=False,left=False,labelright=False)
			plt.savefig(self._dir_lvl4+'/stats_'+indicator+'_'+str(distance)+'_'+name_addon+'.png', bbox_inches='tight',dpi=300, transparent=True); plt.close()

	def plot_stats_hist(self,indicator='ACE'):
		plt.close('all')
		for relative_to_storms in [False,True]:
			fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._nrows*2,self._ncols*2), sharex=True, sharey=True)

			for lab in np.unique(self._clust_labels):
				ax = axes[self._axes_grid[self._grid_labels==lab,0][0],self._axes_grid[self._grid_labels==lab,1][0]]
				ax.annotate(lab, xy=(0,0), xycoords='axes fraction')

				y = self._tracks.loc[(self._tracks['label_lag0']==lab), indicator]
				ax.hist(y, density=True)

				if self._axes_grid[self._grid_labels==lab,0][0] == self._nrows-1:
					ax.set_xlabel(indicator)
				else:
					ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

				if self._axes_grid[self._grid_labels==lab,1][0] == 0:
					ax.set_ylabel('pdf')
				else:
					ax.tick_params(axis='y',which='both',right=False,left=False,labelright=False)
			plt.savefig(self._dir_lvl4+'/stats_'+indicator+'_hist.png', bbox_inches='tight',dpi=300); plt.close()

	def plot_stats_genesisLoc(self,indicator='ACE',distance=2, lag=0, region_colors=None):

		if region_colors is None:
			pie_names = sorted(np.unique(self._tracks.genesis_loc.values))
			pie_colors = ['black','cyan','g','b']
		else:
			pie_names = list(region_colors.keys())
			pie_colors = list(region_colors.values())


		plt.close('all')
		fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._nrows*2,self._ncols*2), sharex=True, sharey=True)

		pie_names[pie_names=='_'] = 'other'
		pies = {}
		for lab in np.unique(self._clust_labels):
			ax = axes[self._axes_grid[self._grid_labels==lab,0][0],self._axes_grid[self._grid_labels==lab,1][0]]
			ax.annotate(lab, xy=(0,0), xycoords='axes fraction')

			pies[lab] = {'pie':[]}
			pies[lab]['radius'] = self._tracks.loc[(self._tracks['label_lag'+str(lag)]==lab), indicator].sum()
			y = []
			for gen_loc,color in zip(pie_names,pie_colors):
				pies[lab]['pie'] += [self._tracks.loc[(self._tracks['label_lag'+str(lag)]==lab) & (self._tracks['genesis_loc']==gen_loc), indicator].sum()]
				y.append(sum(pies[lab]['pie']))

			for gen_loc,color,val in zip(pie_names,pie_colors,y[::-1]):
				ax.bar(x=0,height=val,color=color)

		plt.savefig(self._dir_lvl4+'/stats_origin_'+indicator+'_bar.png', bbox_inches='tight',dpi=300); plt.close()

		max_rad = np.max([val['radius'] for val in pies.values()])

		plt.close('all')
		fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._nrows*2,self._ncols*2), sharex=True, sharey=True)


		for lab in np.unique(self._clust_labels):
			ax = axes[self._axes_grid[self._grid_labels==lab,0][0],self._axes_grid[self._grid_labels==lab,1][0]]
			ax.annotate(lab, xy=(0,0), xycoords='axes fraction')

			wedges, autotexts = ax.pie(pies[lab]['pie'], colors=pie_colors, startangle=90, radius=pies[lab]['radius'])
			ax.set_ylim(-max_rad,max_rad)

		for ax in axes.flatten():
			ax.axis('off')
		axes[int(self._nrows/2),-1].legend(wedges, pie_names,title="origin",loc="lower left",bbox_to_anchor=(1, 0, 0.5, 1))

		plt.savefig(self._dir_lvl4+'/stats_origin_'+indicator+'.png' ,bbox_inches='tight',dpi=300); plt.close()

	def plot_stats_condition(self, indicator, condition, condition_var, cond_details, distance=999, lag=0):
		plt.close('all')
		with PdfPages(self._dir_lvl4+'/stats_'+condition+'_'+indicator+'.pdf') as pdf:
			fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._nrows*2,self._ncols*2), sharex=True, sharey=True)

			labels = [details['name'] for details in cond_details.values()]
			colors = [details['color'] for details in cond_details.values()]
			max_rad = 0
			for lab in np.unique(self._clust_labels):
				ax = axes[self._axes_grid[self._grid_labels==lab,0][0],self._axes_grid[self._grid_labels==lab,1][0]]
				ax.annotate(lab, xy=(0,0), xycoords='axes fraction')

				for i,details in cond_details.items():
					times = self._vector_time[(condition_var>details['l']) & (condition_var<=details['h']) & (self._clust_labels==lab)]
					imp = np.array([self._tracks.loc[(np.array(self._tracks['time'], np.datetime64) == tt) & (self._tracks['distance']<=distance), indicator].sum() for tt in times.values])
					ax.bar(i,imp.sum() / float(len(times)),color=details['color'],label=details['name'])

				if self._axes_grid[self._grid_labels==lab,0][0] == self._nrows-1:
					ax.set_xticks(list(cond_details.keys()))
					ax.set_xticklabels([cond_details[key]['name'] for key in list(cond_details.keys())], rotation=90, fontsize=4)
				else:
					ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

				if self._axes_grid[self._grid_labels==lab,1][0] == 0:
					ax.set_ylabel(indicator)
				else:
					ax.tick_params(axis='y',which='both',right=False,left=False,labelright=False)

			plt.savefig(bbox_inches='tight',dpi=300); plt.close()

	def plot_freq(self):
		plt.close('all')
		fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols+1, figsize=(self._nrows*1,(self._ncols+1)*1), sharex=True, sharey=True, gridspec_kw={'width_ratios':[4]*self._ncols+[1]})
		for i,lab in enumerate(np.unique(self._clust_labels)):
			ax = axes[self._axes_grid[self._grid_labels==lab,0][0],self._axes_grid[self._grid_labels==lab,1][0]]
			ax.annotate(lab, xy=(0,0), xycoords='axes fraction')
			wedges, autotexts = ax.pie(self._stats['freq_mon'].loc[lab].values, colors=self._months['mon_colors'], startangle=90, radius=self._stats['count'].loc[lab].values)
			ax.set_ylim(-max(self._stats['count'].values),max(self._stats['count'].values))
		for ax in axes.flatten():
			ax.axis('off')
		axes[int(self._nrows/2),-1].legend(wedges, self._months['mon_names'],title="Month",loc="lower left",bbox_to_anchor=(1, 0, 0.5, 1))
		plt.savefig(self._dir_lvl4+'/stats_freq_mon.png', transparent=True, bbox_inches='tight',dpi=300); plt.close()

		fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols+1, figsize=(self._nrows*1,(self._ncols+1)*1), sharex=True, sharey=True, gridspec_kw={'width_ratios':[4]*self._ncols+[1]})
		for i,lab in enumerate(np.unique(self._clust_labels)):
			ax = axes[self._axes_grid[self._grid_labels==lab,0][0],self._axes_grid[self._grid_labels==lab,1][0]]
			ax.annotate(lab, xy=(0,0), xycoords='axes fraction')
			wedges, autotexts = ax.pie(self._stats['freq_dec'].loc[lab].values, colors=self._decades['dec_colors'], startangle=90, radius=self._stats['count'].loc[lab].values)
			ax.set_ylim(-max(self._stats['count'].values),max(self._stats['count'].values))
			ax.set_ylim(-max(self._stats['count'].values),max(self._stats['count'].values))
		for ax in axes.flatten():
			ax.axis('off')
		axes[int(self._nrows/2),-1].legend(wedges, self._decades['dec_names'],title="Decade",loc="lower left",bbox_to_anchor=(1, 0, 0.5, 1))
		plt.savefig(self._dir_lvl4+'/stats_freq_dec.png', transparent=True, bbox_inches='tight',dpi=300); plt.close()

	def plot_network(self):
		if 'SOM' in self._clust_add_info.keys():
			with PdfPages(self._dir_lvl4+'/SOM_network.pdf') as pdf:
				fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3)); ax.axis('off')

				nodes = self._clust_centers_map.values.reshape((self._nrows,self._ncols,2))
				ax.scatter(self._mapping[:,0],self._mapping[:,1], s=5, color='lightgray', alpha=0.2)
				for i in range(self._nrows-1):
					for j in range(self._ncols):
						ax.plot([nodes[i,j,0],nodes[i+1,j,0]],[nodes[i,j,1],nodes[i+1,j,1]], color='green')
				for j in range(self._ncols-1):
					for i in range(self._nrows):
						ax.plot([nodes[i,j,0],nodes[i,j+1,0]],[nodes[i,j,1],nodes[i,j+1,1]], color='green')
				im = ax.scatter(self._clust_centers_map[:,0],self._clust_centers_map[:,1], c='green', s=20, zorder=10)

				for lab in np.unique(self._clust_labels):
					im = ax.annotate(lab, xy=(self._clust_centers_map.loc[lab,:][0],self._clust_centers_map.loc[lab,:][1]), zorder=10)

				pdf.savefig(bbox_inches='tight'); plt.close()
		else:
			print('this is not a SOM')

	def plot_network_ax(self, ax):
		nodes = self._clust_centers_map.values.reshape((self._nrows,self._ncols,2))
		ax.scatter(self._mapping[:,0],self._mapping[:,1], s=5, color='lightgray', alpha=0.2)
		for i in range(self._nrows-1):
			for j in range(self._ncols):
				ax.plot([nodes[i,j,0],nodes[i+1,j,0]],[nodes[i,j,1],nodes[i+1,j,1]], color='green')
		for j in range(self._ncols-1):
			for i in range(self._nrows):
				ax.plot([nodes[i,j,0],nodes[i,j+1,0]],[nodes[i,j,1],nodes[i,j+1,1]], color='green')
		im = ax.scatter(self._clust_centers_map[:,0],self._clust_centers_map[:,1], c='green', s=20, zorder=10)

		for lab in np.unique(self._clust_labels):
			im = ax.annotate(lab, xy=(self._clust_centers_map.loc[lab,:][0],self._clust_centers_map.loc[lab,:][1]), zorder=10)

		return ax

	def plot_network_new_sammon(self, ax):
		sys.path.append('/Users/peterpfleiderer/Projects/git-packages/sammon')
		sys.path.append('/p/projects/tumble/carls/shared_folder/git-packages/sammon')
		import sammon; importlib.reload(sammon); from sammon import sammon

		[sammon_proj,E] = sammon(self._clust_centers, 2, display=0, maxiter=1000, maxhalves=1000, init='pca')

		nodes = sammon_proj.reshape((self._nrows,self._ncols,2))

		# plt.close('all')
		# with PdfPages(self._dir_lvl4+'/clustering_network_new_sammon.pdf') as pdf:
		# 	fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3)); ax.axis('off')
		for i in range(self._nrows-1):
			for j in range(self._ncols):
				ax.plot([nodes[i,j,0],nodes[i+1,j,0]],[nodes[i,j,1],nodes[i+1,j,1]], color='green')
		for j in range(self._ncols-1):
			for i in range(self._nrows):
				ax.plot([nodes[i,j,0],nodes[i,j+1,0]],[nodes[i,j,1],nodes[i,j+1,1]], color='green')
		im = ax.scatter(sammon_proj[:,0],sammon_proj[:,1], c='green', s=20, zorder=10)
			# pdf.savefig(bbox_inches='tight'); plt.close()
		return ax

	def plot_label_trend(self, significance=0.05):
		plt.close('all')
		with PdfPages(self._dir_lvl4+'/freq_trend.pdf') as pdf:
			fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=(self._nrows*2,self._ncols*2), sharex=True, sharey=True)

			for lab in np.unique(self._clust_labels):
				ax = axes[self._axes_grid[self._grid_labels==lab,0][0],self._axes_grid[self._grid_labels==lab,1][0]]
				ax.plot(self._stats['freq_year'].loc[lab,:].year, self._stats['freq_year'].loc[lab,:].values)

				if self._stats['freq_trend'].loc[lab,'p_value'] < significance:
					ax.plot(self._stats['freq_year'].loc[lab,:].year, self._stats['freq_year'].loc[lab,:].year*self._stats['freq_trend'].loc[lab,'slope'] + self._stats['freq_trend'].loc[lab,'intercept'], label='p-value %s' %(self._stats['freq_trend'].loc[lab,'p_value'].values.round(3)))
					ax.legend(loc='best', frameon=True, facecolor='w', ncol=1, framealpha=0.6, edgecolor='none', fontsize = 9).set_zorder(1)

				if self._axes_grid[self._grid_labels==lab,0][0] == self._nrows-1:
					ax.set_xlabel('year')
				else:
					ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

				if self._axes_grid[self._grid_labels==lab,1][0] == 0:
					ax.set_ylabel('occurrence [days]')
				else:
					ax.tick_params(axis='y',which='both',right=False,left=False,labelright=False)
			pdf.savefig(bbox_inches='tight'); plt.close()

	def grouped_persistence_get_times(self, tag, group_dict={'name':{'labels':[0,1]}}, max_lag=5):
		self._dir_lvl5 = self._dir_lvl4 + '/' + tag +'/'
		if os.path.isdir(self._dir_lvl5) == False:
			os.system('mkdir '+self._dir_lvl5)



		if len(glob.glob(self._dir_lvl5+'groups_'+tag+'_maxLag*.pkl')) > 0:
			file_name = sorted(glob.glob(self._dir_lvl5+'groups_'+tag+'_maxLag*.pkl'))[-1]
			with open(file_name, 'rb') as infile:
				group_dict = pickle.load(infile)

		file_name = self._dir_lvl5+'groups_'+tag+'_maxLag'+str(max_lag)+'.pkl'

		# group_dict = {
		# 	# 'hamper VWS' : {'labels':[12,16,17,18], 'position':[0,0]},
		# 	'hamper MSLP' : {'labels':[0,1,4,8], 'position':[0,1]},
		# 	# 'favorable' : {'labels':[10,11,14,15], 'position':[1,0]},
		# 	# 'other' : {'labels':[2,3,5,6,7,9,13,19], 'position':[1,1]},
		# }
		# tag='test'
		# max_lag=5


		for name,details in group_dict.items():
			times = np.array(self._vector_time[np.isin(self._clust_labels,details['labels'])], np.datetime64)
			details['times_lag0'] = times
			for lag in range(1,max_lag):
				if 'times_lag'+str(lag) not in details.keys():

					persist,new = [],[]
					for it,tt in enumerate(times):
						if tt - np.timedelta64(lag,'D') in self._vector_time:
							if self._clust_labels[self._vector_time == tt - np.timedelta64(lag,'D')] in details['labels']:
								persist.append(it)
							else:
								new.append(it)
						else:
							new.append(it)

					details['times_lag'+str(lag)] = np.delete(times,persist,0)
					times = np.delete(times,new,0)

		with open(file_name, 'wb') as outfile:
			pickle.dump(group_dict, outfile)

		self._groups = group_dict

		track_times = np.array(self._tracks['time'], np.datetime64)
		for lag in range(0,max_lag):
			self._tracks['group_lag'+str(lag)] = '-'
			for name,details in group_dict.items():
				times = self._groups[name]['times_lag'+str(lag)]
				self._tracks.loc[np.isin(track_times,times), 'group_lag'+str(lag)] = name

		self._tracks.to_csv(self._dir_lvl4+'/stats/stats_TC.csv')

	def grouped_persistence_indicator_averages(self, indicators, distances):
		file_name = self._dir_lvl5+'group_stats.pkl'
		lag_times = np.array([xx for xx in self._groups[list(self._groups.keys())[0]].keys() if 'lag' in xx])

		self._group_stats = {}
		if os.path.isfile(file_name):
			with open(file_name, 'rb') as infile:
				self._group_stats = pickle.load(infile)

		else:
			self._group_stats['count'] = {}
			for lag_time in lag_times:
				self._group_stats['count'][lag_time] = {}
				for name,details in self._groups.items():
					self._group_stats['count'][lag_time][name] = float(len(details[lag_time]))


		for indicator in indicators:
			if indicator not in self._group_stats:
				self._group_stats[indicator] = {}
			for distance in distances:
				if indicator not in self._group_stats[indicator].keys():
					self._group_stats[indicator][distance] = {}
				for lag_time in lag_times:
					if lag_time not in self._group_stats[indicator][distance].keys():
						self._group_stats[indicator][distance][lag_time] = {}

		for lag_time in lag_times:
			for name,details in self._groups.items():
				for distance in distances:
					tmp = self._tracks.loc[np.isin(np.array(self._tracks['time'], np.datetime64),details[lag_time]) & (self._tracks['distance']<=distance)]
					for indicator in indicators:
						self._group_stats[indicator][distance][lag_time][name] = np.sum(tmp.loc[:,indicator]) / float(len(details[lag_time]))

		with open(file_name, 'wb') as outfile:
			pickle.dump(self._group_stats, outfile)

	def plot_grouped_persistence_stats(self, indicator, distance_colors):
		plt.close('all')
		with PdfPages(self._dir_lvl5+'/stats_grouped_'+indicator+'_'+'d'.join([str(dd) for dd in list(distance_colors.keys())]) +'.pdf') as pdf:
			nrows,ncols = tuple(np.array([xx['position'] for xx in self._groups.values()]).max(axis=0)+1)
			for lag_time in np.array([xx for xx in self._groups[list(self._groups.keys())[0]].keys() if 'lag' in xx]):
				fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(self._nrows*1,(self._ncols)*1), sharex=True, sharey=True)
				max_rad = 0
				for name,details in self._groups.items():
					ax = axes[details['position'][0],details['position'][1]]
					ax.annotate(name, xy=(0,0), xycoords='axes fraction')
					for distance,color in distance_colors.items():
						wedges, autotexts = ax.pie([1], radius=self._group_stats[indicator][distance][lag_time][name], colors=[color])
						max_rad = max([max_rad,self._group_stats[indicator][distance][lag_time][name]])

				ax.set_ylim(-max_rad,max_rad)
				for ax in axes.flatten():
					ax.axis('off')

				plt.suptitle(lag_time.split('_')[-1])
				pdf.savefig(bbox_inches='tight'); plt.close()

	def plot_grouped_persistence_stats_bar(self, indicator, distance_colors):
		plt.close('all')
		with PdfPages(self._dir_lvl5+'/stats_grouped_'+indicator+'_'+'d'.join([str(dd) for dd in list(distance_colors.keys())]) +'_bar.pdf') as pdf:
			nrows,ncols = tuple(np.array([xx['position'] for xx in self._groups.values()]).max(axis=0)+1)
			fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3,nrows*3), sharex=True, sharey=True)

			for name,details in self._groups.items():
				ax = axes[details['position'][0],details['position'][1]]
				ax.set_title(name)
				xlabels = {}
				for i,lag_time in enumerate([xx for xx in self._groups[list(self._groups.keys())[0]].keys() if 'lag' in xx]):

					for distance,color in distance_colors.items():
						ax.bar(i, self._group_stats[indicator][distance][lag_time][name], color=color, width=0.8)
					xlabels[i] = lag_time.split('lag')[-1]

				if details['position'][0] == nrows-1:
					ax.set_xticks(sorted(list(xlabels.keys())))
					ax.set_xticklabels([xlabels[key] for key in sorted(list(xlabels.keys()))])
					ax.set_xlabel('lag')

				if details['position'][1] == 0:
					ax.set_ylabel(indicator)

			pdf.savefig(bbox_inches='tight'); plt.close()

	def plot_grouped_frequency(self):
		def func(pct, allvals):
			absolute = int(pct/100.*np.sum(allvals))
			return "{:d}".format(absolute)

		plt.close('all')
		with PdfPages(self._dir_lvl5+'/stats_freq_grouped.pdf') as pdf:
			nrows,ncols = tuple(np.array([xx['position'] for xx in self._groups.values()]).max(axis=0)+1)
			fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3,nrows*3), sharex=True, sharey=True)
			max_rad = 0
			for name,details in self._groups.items():
				ax = axes[details['position'][0],details['position'][1]]
				ax.set_title(name)
				lag_times = [xx for xx in self._groups[list(self._groups.keys())[0]].keys() if 'lag' in xx and xx != 'times_lag0']
				labels = [xx.split('lag')[-1] for xx in lag_times]
				wedges = [self._group_stats['count'][xx][name] for xx in lag_times]
				radius = self._group_stats['count']['times_lag0'][name]
				ax.pie(wedges, labels=labels, autopct=lambda pct: func(pct, wedges), radius=radius)

				max_rad = max([max_rad,radius])

				ax.set_ylim(-max_rad,max_rad)

			for ax in axes.flatten():
				ax.axis('off')

			pdf.savefig(bbox_inches='tight'); plt.close()

	####################
	# check plots
	####################

	def check_field_characteristics(self):
		plt.close('all')
		with PdfPages(self._dir_lvl0+'/fields_variability.pdf') as pdf:
			for field_name in np.unique(self._vector_var):
				tmp_field = self._vector[:,(self._vector_var==field_name)].values.reshape(self._vector.shape[0],len(self._lat),len(self._lon))

				# Mean
				fig,ax = plt.subplots(nrows=1, figsize=(6,6*(len(self._lat)/len(self._lon))), subplot_kw={'projection': ccrs.PlateCarree()})
				ax.annotate(field_name+' mean', xy=(0,0), xycoords='axes fraction', backgroundcolor='w')
				ax.coastlines(); ax.set_extent([self._lon.min(),self._lon.max(),self._lat.min(),self._lat.max()], crs=ccrs.PlateCarree())
				z = tmp_field.mean(0)
				limit = np.max(np.abs(np.nanpercentile(z.flatten(), [5,95])))
				ax.contourf(self._lon,self._lat, z, vmin=-limit, vmax=limit, cmap=self._input_data_details[field_name]['cmap'])
				pdf.savefig(bbox_inches='tight'); plt.close()

				im = plt.scatter(np.linspace(-limit,limit,10),np.linspace(-limit,limit,10), c=np.linspace(-limit,limit,10), cmap=self._input_data_details[field_name]['cmap'])
				fig,ax = plt.subplots(nrows=1, figsize=(2,0.5))
				cb = fig.colorbar(im, cax=ax, orientation='horizontal',label=field_name+' [sigma]')
				pdf.savefig(bbox_inches='tight'); plt.close()

				# STD
				fig,ax = plt.subplots(nrows=1, figsize=(6,6*(len(self._lat)/len(self._lon))), subplot_kw={'projection': ccrs.PlateCarree()})
				ax.annotate(field_name+' std', xy=(0,0), xycoords='axes fraction', backgroundcolor='w')
				ax.coastlines(); ax.set_extent([self._lon.min(),self._lon.max(),self._lat.min(),self._lat.max()], crs=ccrs.PlateCarree())
				z = tmp_field.std(0)
				bounds = np.nanpercentile(z.flatten(), [5,95])
				ax.contourf(self._lon,self._lat,z, vmin=bounds[0], vmax=bounds[1], cmap=self._input_data_details[field_name]['cmap'])
				pdf.savefig(bbox_inches='tight'); plt.close()

				im = plt.scatter(np.linspace(bounds[0],bounds[1],10),np.linspace(bounds[0],bounds[1],10), c=np.linspace(bounds[0],bounds[1],10), cmap=self._input_data_details[field_name]['cmap'])
				fig,ax = plt.subplots(nrows=1, figsize=(2,0.5))
				cb = fig.colorbar(im, cax=ax, orientation='horizontal',label=field_name+' [sigma]')
				pdf.savefig(bbox_inches='tight'); plt.close()

				# Trend
				fig,ax = plt.subplots(nrows=1, figsize=(6,6*(len(self._lat)/len(self._lon))), subplot_kw={'projection': ccrs.PlateCarree()})
				ax.annotate(field_name+' trend', xy=(0,0), xycoords='axes fraction', backgroundcolor='w')
				ax.coastlines(); ax.set_extent([self._lon.min(),self._lon.max(),self._lat.min(),self._lat.max()], crs=ccrs.PlateCarree())

				regr = sklearn.linear_model.LinearRegression()
				slope = z.copy()*np.nan
				p_values = z.copy()*np.nan
				for iy,y in enumerate(self._lat.values):
					for ix,x in enumerate(self._lon.values):
						slope[iy,ix], intercept, r_value, p_values[iy,ix], std_err = scipy.stats.linregress(self._vector_time.dt.year.values,tmp_field[:,iy,ix])
				limit = np.max(np.abs(np.nanpercentile(slope.flatten(), [5,95])))
				im = ax.contourf(self._lon,self._lat, slope, vmin=-limit, vmax=limit, cmap=self._input_data_details[field_name]['cmap'])
				ax.contourf(self._lon,self._lat, p_values, levels=[0.01,1], hatches=['////'], colors=['none'])
				pdf.savefig(bbox_inches='tight'); plt.close()

				im = plt.scatter(np.linspace(-limit,limit,10),np.linspace(-limit,limit,10), c=np.linspace(-limit,limit,10), cmap=self._input_data_details[field_name]['cmap'])
				fig,ax = plt.subplots(nrows=1, figsize=(2,0.5))
				cb = fig.colorbar(im, cax=ax, orientation='horizontal',label=field_name+' [sigma]')
				pdf.savefig(bbox_inches='tight'); plt.close()

	# def plot_group_heatmap(self, group_dict, indicator, distance, **kwargs):
	# 	plt.close('all')
	# 	with PdfPages(self._dir_lvl4+'/stats_'+indicator+'_'+str(distance)+'_kde_grouped.pdf') as pdf:
	#
	# 		for lag_time in np.array([xx for xx in group_dict[list(group_dict.keys())[0]].keys() if 'lag' in xx]):
	# 			tmp = np.zeros(np.array([xx['position'] for xx in group_dict.values()]).max(axis=0)+1)
	# 			for name,details in group_dict.items():
	# 				tmp[details['position'][0],details['position'][1]] = np.mean([self._tracks.loc[np.array(self._tracks['time'], np.datetime64) == tt,indicator].sum() for tt in details[lag_time]])
	#
	# 			plt.imshow(tmp,aspect='auto', **kwargs)
	# 			plt.axis('off')
	# 			for name,details in group_dict.items():
	# 				plt.annotate(name, xy=details['position'][::-1], ha='center', va='center')
	# 			plt.suptitle(lag_time.split('_')[-1])
	# 			pdf.savefig(bbox_inches='tight'); plt.close()
	#
	# 		tmp[0,0] = -100
	# 		tmp[-1,0] = 100
	# 		im = plt.imshow(tmp, aspect='auto', **kwargs)
	#
	# 		fig,ax = plt.subplots(nrows=1, figsize=(2,0.5))
	# 		cb = fig.colorbar(im, cax=ax, orientation='horizontal',label=indicator)
	# 		pdf.savefig(bbox_inches='tight'); plt.close()

	# def load_TC_details(self, file_name, wind_var_name='wmo_wind'):
	# 	TC_details = xr.open_dataset(file_name)
	# 	notna1 = (TC_details['lon'] >= self._lon.min()) & (TC_details['lon'] <= self._lon.max())
	# 	notna2 = (TC_details['lat'] >= self._lat.min()) & (TC_details['lat'] <= self._lat.max())
	# 	notna = notna1 & notna2
	# 	TC_details['wind'] = TC_details[wind_var_name]
	# 	for var in ['lon','lat','time','wind','distance','storm_time']:
	# 		TC_details[var] = TC_details[var][notna]
	# 	TC_details = TC_details.assign_coords(time=np.array([str(tt)[:10] for tt in TC_details.time.values],'datetime64[ns]'))
	# 	self._TC_details = TC_details
	#
	# def load_TC_stats(self, file_name):
	# 	TC_stats = xr.open_dataset(file_name)['allTCs']
	# 	TC_stats.values[np.isnan(TC_stats)] = 0
	# 	TC_stats = TC_stats.assign_coords(time=np.array([str(tt)[:10] for tt in TC_stats.time.values],'datetime64[ns]'))
	# 	self._TC_stats = TC_stats
	#
	# 	maj = self._TC_stats.loc[:,:,['maxWind']].copy()
	# 	maj.values[maj<=112] = 0
	# 	maj.values[maj>112] = 1
	# 	maj = maj.assign_coords(type=['major'])
	#
	# 	self._TC_stats = xr.concat((self._TC_stats,maj), dim='type')


	# def plot_grouped_frequency(self):
	# 	plt.close('all')
	# 	with PdfPages(self._dir_lvl5+'/stats_freq_grouped.pdf') as pdf:
	# 		nrows,ncols = tuple(np.array([xx['position'] for xx in self._groups.values()]).max(axis=0)+1)
	# 		for lag_time in np.array([xx for xx in self._groups[list(self._groups.keys())[0]].keys() if 'lag' in xx]):
	# 			fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(self._nrows*1,(self._ncols)*1), sharex=True, sharey=True)
	# 			max_rad = 0
	# 			for name,details in self._groups.items():
	# 				ax = axes[details['position'][0],details['position'][1]]
	# 				ax.annotate(name, xy=(0,0), xycoords='axes fraction')
	# 				wedges, autotexts = ax.pie([1], radius=len(details[lag_time]))
	# 				#print(lag_time,name,len(details[lag_time]))
	# 				max_rad = max([max_rad,len(details[lag_time])])
	#
	# 			ax.set_ylim(-max_rad,max_rad)
	# 			for ax in axes.flatten():
	# 				ax.axis('off')
	#
	# 			plt.suptitle(lag_time.split('_')[-1])
	# 			pdf.savefig(bbox_inches='tight'); plt.close()
	#



	# def plot_stats_condition(self, condition, indicator='ACE', lag=0, cond_details={0:{'name':'False','color':'c'},1:{'name':'True','color':'m'}}):
	#
	# 	cond_colors = [cond_details[k]['color'] for k in sorted(cond_details.keys())]
	# 	cond_names = [cond_details[k]['name'] for k in sorted(cond_details.keys())]
	# 	cond_vals = [k for k in sorted(cond_details.keys())]
	#
	# 	pies = {}
	# 	for lab in np.unique(self._clust_labels):
	# 		pies[lab] = {'pie':[]}
	# 		pies[lab]['radius'] = self._tracks.loc[(self._tracks['label_lag'+str(lag)]==lab), indicator].sum()
	# 		for val in cond_vals:
	# 			pies[lab]['pie'] += [self._tracks.loc[(self._tracks['label_lag'+str(lag)]==lab) & (self._tracks[condition]==val), indicator].sum()]
	#
	# 	max_rad = np.max([val['radius'] for val in pies.values()])
	#
	# 	plt.close('all')
	# 	with PdfPages(self._dir_lvl4+'/stats_'+condition+'_'+indicator+'.pdf') as pdf:
	# 		fig,axes = plt.subplots(nrows=self._nrows, ncols=self._ncols+1, figsize=(self._nrows*2,self._ncols*2), sharex=True, sharey=True)
	#
	#
	# 		for lab in np.unique(self._clust_labels):
	# 			ax = axes[self._axes_grid[self._grid_labels==lab,0][0],self._axes_grid[self._grid_labels==lab,1][0]]
	# 			ax.annotate(lab, xy=(0,0), xycoords='axes fraction')
	#
	# 			wedges, autotexts = ax.pie(pies[lab]['pie'], colors=cond_colors, startangle=90, radius=pies[lab]['radius'])
	# 			ax.set_ylim(-max_rad,max_rad)
	#
	# 		for ax in axes.flatten():
	# 			ax.axis('off')
	# 		axes[int(self._nrows/2),-1].legend(wedges, cond_names, loc="center")
	#
	# 		pdf.savefig(bbox_inches='tight'); plt.close()

#
