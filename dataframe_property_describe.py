import numpy as np
import pandas as pd
import concurrent.futures
import math
import copy
from itertools import repeat

class dataframe_optimizer:
	def __init__(self, **kwargs):
		self.reset(**kwargs)
		
	def reset(self, **kwargs):
		self.input_params = {};
		self.input_params['filename'] = kwargs.pop('filename')
		self.input_params['encoding'] = kwargs.pop('encoding')	
		
		if self.input_params['filename'] is None or self.input_params['encoding'] is None:
			raise ValueError('filename and encoding must be provided at least')
			
		df = self.get_peek_dataframe()
		print("Peek dataframe")
		print(df)
		
		self.input_params['usecols'] = list(df.columns.values)
		
		del_cols = kwargs.pop('del_cols')
		if del_cols is not None:
			for col in del_cols:
				self.input_params['usecols'].remove(col)
		
		self.input_params['parse_dates'] = kwargs.pop('parse_dates')
		if to_date_cols is not None:
			for col in to_date_cols:
				if col not in self.input_params['usecols']:
					self.input_params['parse_dates'].remove(col)
					
		self.input_params['numerical_dtype'] = {}
		self.input_params['cat_dtype'] = []	
		
		
	def get_peek_dataframe(self):
		df = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			nrows = 5)
		return df
		
	def optimize(self, show_info = False, chunksize = None, parallel = False):
		df_reader = None
		
		if chunksize is None:
			df_reader = dataframe_reader(self.input_params, parallel)
		else
			df_reader = dataframe_chunk_reader(self.input_params, chunksize, parallel)
			

			
class base_dataframe_reader:
	def __init__(self):
		pass
	
	def get_optimal_df_params(self):
		pass
		
		
class dataframe_reader(base_dataframe_reader):
	def __init__(self, input_params, parallel):
		super(dataframe_reader, self).__init__()
		self.input_params = input_params
		self.parallel = parallel
	
	def get_optimal_df_params(self)
		self.df = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			usecols = self.input_params['usecols'],
			parse_dates = self.input_params['parse_dates'],
			dtype = self.input_params['cat_dtype'])
		
		# sw workaround for float->int specification
		for col, type_str in self.input_params['numerical_dtype'].items():
			self.df[col] = self.df[col].astype(type_str)
			
		return dataframe_reader.parse(df)
		
	@staticmethod
	def parse(df):
	
		ret_dict = {}
		ret_dict['memory_footprints'] = df.memory_usage(deep=True).sum()
		ret_dict['total_rows'] = len(df)
		ret_dict['null_counts'] = df.select_dtypes(include=['float']).isnull().sum()
		
		ret_dict['numerical_dtype'] = {}
		
		float_cols = df.select_dtypes(include=['float'])
		for col in float_cols.columns:
			if data_info['null_counts'][col] != 0:
				ret_dict['numerical_dtype'][col] = pd.to_numeric(df[col], downcast='float').dtype
			else:
				ret_dict['numerical_dtype'][col] = pd.to_numeric(df[col].astype('int'), downcast='integer').dtype
	
			ret_dict['numerical_dtype'][col] = str(ret_dict['numerical_dtype'][col])	
		
		ret_dict['numerical_dtype'] = {}
		
		obj_cols = df.select_dtypes(include=['object'])
		for col in obj_cols.columns:
			unique_values[col] = df[col].unique()
		
		
		return ret_dict
		
	
	
class dataframe_chunk_reader(dataframe_reader):
	def __init__(self, input_params, chunksize, parallel):
		super(dataframe_chunk_reader, self).__init__(input_params, parallel)
		self.chunksize = chunksize
	
	def get_optimal_df_params(self)
		chunk_iter = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			chunksize = self.chunksize,
			usecols = self.input_params['usecols'],
			parse_dates = self.input_params['parse_dates'],
			dtype = self.input_params['cat_dtype'])
		
		for chunk in chunk_iter:
			# sw workaround for float->int specification
			for col, type_str in self.input_params['numerical_dtype'].items():
				chunk[col] = chunk[col].astype(type_str)
				
			dataframe_reader.parse(chunk)
		
				


