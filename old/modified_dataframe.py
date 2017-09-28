import numpy as np
import pandas as pd
import concurrent.futures
import math
import copy
from itertools import repeat

from modified_dataframe_ops import dataframe_ops
from local_util import local_util

'''
dataframe is use polymorphismed into 
normal dataframe and chunk dataframe
'''
class dataframe_base:
	def __init__(self):
		pass

	def perform_action(self, func, *args):
		pass

	
	def collect_info(self):
		# reset self.data_info
		self.data_info = {}
		
		# collect return data info
		data_info_list = self.perform_action(dataframe_ops.collect)
		data_info_list = local_util.make_list_dict(data_info_list)
		
		self.data_info['memory_footprints'] = np.sum([data_info['memory_footprints'] for data_info in data_info_list])
		self.data_info['total_rows'] = np.sum([data_info['total_rows'] for data_info in data_info_list])

		# data_info['null_counts'] becomes a dictionary
		null_count_combined = pd.concat([data_info['null_counts'] for data_info in data_info_list])
		self.data_info['null_counts'] = null_count_combined.groupby(null_count_combined.index).sum()
		
	
	def get_info(self, keyword=None):
		if keyword is None:
			return self.data_info
		elif keyword in self.data_info:
			return self.data_info[keyword]
		else:
			raise ValueError("If keyword is provided, it should be contained in data_info")

		
	def show_data_info(self):
		print('\nMemory usage: %.2f (MB)' % (self.data_info['memory_footprints']/(1024*1024)))
		print('Total rows: ', self.data_info['total_rows']);
		
		print('Null counts:')
		for k,v in self.data_info['null_counts'].items():
			print(' ', k, ' '*(24-len(k)), v)
	
	
class normal_dataframe(dataframe_base):
	def __init__(self, input_params, parallel = None):
		super(normal_dataframe, self).__init__()
		self.input_params = input_params
		
		self.df = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			chunksize = self.input_params['chunksize'],
			usecols = self.input_params['usecols'],
			parse_dates = self.input_params['parse_dates'],
			dtype = self.input_params['dtype'])
	
		# sw workaround for float->int specification
		for col, type_str in self.input_params['numerical_dtypes'].items():
			self.df[col] = self.df[col].astype(type_str)
		self.parallel = parallel
		print("Memory footprint is not accurate in the parallel veresion due to dataframe splitting mechanism!")
		
		# collect info use the information of input params, df, parallel
		# hence must be put in the last 
		self.collect_info()
			
		
	def perform_action(self, func, *args):
		if self.parallel is True:
		
			with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
				separate_dfs = np.array_split(self.df, 8)
				packed_args = ((func, df, self.data_info, *args) for df in separate_dfs)
				res = executor.map(self.pack_subprocess, packed_args)
				return res
			return None
		else:
			return func(self.df, self.data_info, *args)
			
	def pack_subprocess(self, packed_input):
		if len(packed_input) > 3:
			return packed_input[0](packed_input[1], packed_input[2], packed_input[3])
		else:
			return packed_input[0](packed_input[1], packed_input[2])
			
class chunk_dataframe(dataframe_base):
	def __init__(self, input_params, parallel = None):
		super(chunk_dataframe, self).__init__()
		self.input_params = input_params
		self.collect_info()
		self.parallel = parallel
		
	def perform_action(self, func, *args):
		chunk_iter = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			chunksize = self.input_params['chunksize'],
			usecols = self.input_params['usecols'],
			parse_dates = self.input_params['parse_dates'],
			dtype = self.input_params['dtype'])
			
		result = []	
		for chunk in chunk_iter:
			result.append(self.chunk_process(func, chunk, *args))
		
		return result
				
	def chunk_process(self, func, chunk, *args):
		# sw workaround for float->int specification
		for col, type_str in self.input_params['numerical_dtypes'].items():
			chunk[col] = chunk[col].astype(type_str)
		return func(chunk, self.data_info, *args)

		
class parallel_chunk_dataframe(chunk_dataframe):
	def __init__(self, input_params):
		super(parallel_chunk_dataframe, self).__init__(input_params)
		
		
	def perform_action(self, func, *args):
		chunk_iter = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			chunksize = self.input_params['chunksize'],
			usecols = self.input_params['usecols'],
			parse_dates = self.input_params['parse_dates'],
			dtype = self.input_params['dtype'])
		
		with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
			packed_args = ((func, chunk, *args) for chunk in chunk_iter)
			
			res = executor.map(self.pack_subprocess, packed_args)
				
			'''
			res = []
			for future in (executor.submit(self.subprocess, func, chunk, *args) for chunk in chunk_iter):
				res.append(future.result())
			
			futures = [executor.submit(self.subprocess, func, chunk, *args) for chunk in chunk_iter]
			
			# This print may be slightly delayed, as futures start executing as soon as the pool begins to fill,
			# eating your CPU time
			print("Executing total", len(futures), "jobs")

			res = []
			# Wait the executor to complete each future, give 180 seconds for each job
			for idx, future in enumerate(concurrent.futures.as_completed(futures, timeout=180.0)):
				res.append(future.result())  # This will also raise any exceptions
				percentage = (idx+1)/len(futures)*100.0
				percentage_int = int(percentage/10)
				print("\rJob status: {}{} ({:3.2f}%)".format(chr(0x2588)*(percentage_int),chr(0x25A0)*(10-percentage_int),percentage), end="")
			print('')
			'''

			return res
		
		return []
	
	def pack_subprocess(self, packed_input):
		if len(packed_input) > 2:
			return self.subprocess(packed_input[0], packed_input[1], packed_input[2])
		else:
			return self.subprocess(packed_input[0], packed_input[1])
'''
An uniform interface to get normal dataframe or chunk dataframe from  
'''		
class dataframe_reader:
	def __init__(self, input_params, parallel = False):
		self.input_params = input_params
		self.parallel = parallel
		
	def get_dataframe(self):		
		if self.input_params['chunksize'] is None:
			return normal_dataframe(self.input_params, self.parallel)
		else:
			return chunk_dataframe(self.input_params, self.parallel)