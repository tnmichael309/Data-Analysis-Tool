import numpy as np
import pandas as pd
import concurrent.futures
import math
import copy
from itertools import repeat
import pickle

from dataframe_property_handler import dataframe_property_handler
from dataframe_parallel_manager import df_parall_manager

class dataframe_optimizer:
	def __init__(self, **kwargs):
		self.reset(**kwargs)
		
	def reset(self, **kwargs):
		self.input_params = {};
		
		if 'filename' not in kwargs or 'encoding' not in kwargs:
			raise ValueError('Filename and Encoding must be provided!')
			
		self.input_params['filename'] = kwargs.pop('filename')
		self.input_params['encoding'] = kwargs.pop('encoding')	
	
		df = self.get_peek_dataframe()
		
		self.input_params['usecols'] = list(df.columns.values)
		
		if 'del_cols' in kwargs:
			del_cols = kwargs.pop('del_cols')
			for col in del_cols:
				self.input_params['usecols'].remove(col)
		
		self.input_params['parse_dates'] = []
		if 'parse_dates' in kwargs:
			to_date_cols = kwargs.pop('parse_dates')
			for col in to_date_cols:
				if col in self.input_params['usecols']:
					self.input_params['parse_dates'].append(col)
					
		self.input_params['numerical_dtype'] = {}
		self.input_params['cat_dtype'] = {}	
		
		
	def get_peek_dataframe(self, nrows=5):
		print("Peek dataframe: ", self.input_params['filename'])
		print("Input params: \n", self.input_params)
		df = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			nrows = nrows)
		print(df)
		return df
		
	def optimize(self, show_info = False, chunksize = None, parallel = False):
		df_reader = None
		
		if chunksize is None:
			df_reader = dataframe_reader(self.input_params, parallel)
		else:
			df_reader = dataframe_chunk_reader(self.input_params, chunksize, parallel)
		
		print('\n==========================================================================')
		print('Phase 1: Parse dataframe info for compact numerical type and category type')
		print('==========================================================================')
		print("Input params: \n", self.input_params)
		info = df_reader.parse_df_information()
		if show_info is True:
			dataframe_property_handler.print_info(info)
			
		new_input_params = copy.deepcopy(self.input_params)
		
		col_info = info['col_info']
		for col in col_info:
			if col_info[col]['dtype'] == 'object' and col_info[col]['is_to_cat'] is True:
				new_input_params['cat_dtype'][col] = 'category'
				
			if col_info[col]['dtype'] == 'float64':
				new_input_params['numerical_dtype'][col] = col_info[col]['numerical_dtype']
				
		print('\n==========================================================================')
		print('Phase 2: Optimize input parameters and Parse the dataframe with new update')
		print('==========================================================================')
		print("New Input params: \n", new_input_params)
		df_reader.reset_param(new_input_params, parallel)
		info = df_reader.parse_df_information()
		if show_info is True:
			dataframe_property_handler.print_info(info)
			
		return new_input_params

	@staticmethod
	def dump_dataframe_params(input_params, filename):
		pickle.dump(input_params, open(filename,'wb'))
		print('Dataframe info is dumped into file: ', filename)				
	
	@staticmethod	
	def load_dataframe_params(filename):
		input_params = pickle.load(open(filename, 'rb'))
		print('Dataframe info is loaded from file: ', filename)						
		return input_params
	
	
class base_dataframe_reader:
	def __init__(self):
		pass
	
	def get_optimal_df_params(self):
		pass
		
		
class dataframe_reader(base_dataframe_reader):
	def __init__(self, input_params, parallel):
		super(dataframe_reader, self).__init__()
		self.reset_param(input_params, parallel)
	
	def reset_param(self, input_params, parallel):
		self.input_params = input_params
		self.parallel = parallel
		
		if self.parallel is True:
			print("Using CPU # = ", df_parall_manager.get_cpu_count())
			
	def get_dataframe(self):
		df = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			usecols = self.input_params['usecols'],
			parse_dates = self.input_params['parse_dates'],
			dtype = self.input_params['cat_dtype'])
		
		# sw workaround for float->int specification
		for col, type_str in self.input_params['numerical_dtype'].items():
			df[col] = df[col].astype(type_str)
			
		return df
	
	# parse what need to be run all over the data with
	def get_df_basic_information(self, df):
		ret = dataframe_property_handler.get_empty_df_information_dict()
		ret['total_row'] = df.shape[0]
		ret['total_mem'] = df.memory_usage(deep=True).sum()
		
		null_counts_info = df.select_dtypes(include=['float']).isnull().sum()
		return ret, null_counts_info
		
	
	def get_df_basic_information_parallel(self, df):
	
		df_split = np.array_split(df, df_parall_manager.get_cpu_count())
		
		ret = dataframe_property_handler.get_empty_df_information_dict()
		with concurrent.futures.ProcessPoolExecutor(max_workers=df_parall_manager.get_cpu_count()) as executor:
			res = executor.map(self.get_df_basic_information, df_split)
			res = list(res)
			
			ret['total_row'] = np.sum([r[0]['total_row'] for r in res])
			ret['total_mem'] = np.sum([r[0]['total_mem'] for r in res])
			null_count_combined = pd.concat([r[1] for r in res])
			null_counts_info = null_count_combined.groupby(null_count_combined.index).sum()
			
			return ret, null_counts_info
		print('Multiprocessing failed')	
		return None, None
		
	# interface for call from outside
	def parse_df_information(self):
		df = self.get_dataframe()
		return self.process_df_information(df)
		
	def process_df_information(self, df):
	
		if self.parallel == False:
			ret, null_counts_info = self.get_df_basic_information(df) 
			
			for col in df.columns:	
				ret['col_info'][col] = self.process_col_information(df, col, null_counts_info, ret['total_row'])
			return ret
		else:
			ret, null_counts_info = self.get_df_basic_information_parallel(df) 
			with concurrent.futures.ProcessPoolExecutor(max_workers=df_parall_manager.get_cpu_count()) as executor:
				packed_args = ((df, col, null_counts_info, ret['total_row']) for col in df.columns)
				res = list(executor.map(self.process_col_information_packed, packed_args, chunksize = 20000))
				for i, col in enumerate(df.columns):
					ret['col_info'][col] = res[i]
				
				return ret
			
			print('Multiprocessing failed')
			return ret
			
	
	def process_col_information_packed(self, args):
		return self.process_col_information(args[0], args[1], args[2], args[3])
		
	def process_col_information(self, df, col, null_counts_info, total_rows):

		temp_col_info = {}
		temp_col_info['dtype'] = df[col].dtype.name # current data type

		if col in null_counts_info:
			if null_counts_info[col] == 0: # no null value, can be tranformed to int, then cast to most compact type
				temp_col_info['numerical_dtype'] = pd.to_numeric(df[col].astype('int'), downcast='integer').dtype.name
			else:
				temp_col_info['numerical_dtype'] = pd.to_numeric(df[col], downcast='float').dtype.name
		else:
			temp_col_info['numerical_dtype'] = None
	
		if df[col].dtype.name == 'object':
			unique_vals = df[col].unique()
			if len(unique_vals) / total_rows >= .5:
				temp_col_info['is_to_cat'] = False
			else:
				temp_col_info['is_to_cat'] = True
		else:
			temp_col_info['is_to_cat'] = None
		
		return temp_col_info
		
class dataframe_chunk_reader(dataframe_reader):
	def __init__(self, input_params, chunksize, parallel):
		super(dataframe_chunk_reader, self).__init__(input_params, parallel)
		self.chunksize = chunksize
	
	def get_dataframe(self):
		chunk_iter = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			chunksize = self.chunksize,
			usecols = self.input_params['usecols'],
			parse_dates = self.input_params['parse_dates'],
			dtype = self.input_params['cat_dtype'])
		
		return chunk_iter
		
	# when we try to use chunk to read, 
	# it meas the file is really big
	# parallelization maynot suit in the memory_usage
	# and event reading a file, we cannot use multiprocessing
	def parse_df_information(self):
		chunk_iter = self.get_dataframe()
		
		res = []
		for chunk in chunk_iter:
			# sw workaround for float->int specification
			for col, type_str in self.input_params['numerical_dtype'].items():
				chunk[col] = chunk[col].astype(type_str)
				
			res.append(self.process_df_information(chunk))
		
		# merge result and return
		return dataframe_property_handler.merge_df_information_dict(res)
		
	def process_col_information(self, df, col, null_counts_info, total_rows):

		temp_col_info = {}
		temp_col_info['dtype'] = df[col].dtype.name # current data type

		if col in null_counts_info:
			if null_counts_info[col] == 0: # no null value, can be tranformed to int, then cast to most compact type
				temp_col_info['numerical_dtype'] = pd.to_numeric(df[col].astype('int'), downcast='integer').dtype.name
			else:
				temp_col_info['numerical_dtype'] = pd.to_numeric(df[col], downcast='float').dtype.name
		else:
			temp_col_info['numerical_dtype'] = None
	
		if df[col].dtype.name == 'object':
			temp_col_info['is_to_cat'] = df[col].unique()
		else:
			temp_col_info['is_to_cat'] = None
			
		return temp_col_info 