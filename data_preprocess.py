import numpy as np
import pandas as pd
import copy

# lib for modified df
from modified_dataframe import dataframe_reader
from modified_dataframe_ops import dataframe_ops
from local_util import local_util
	
class data_preprocessor:
	def __init__(self, filename,
		# optional inputs
		encoding = 'utf-8',
		chunksize = None,
		parallel = False):
		
		# need to stored in the object in order to 
		# further optimize other input params settings		
		self.input_params = {};
		self.input_params['filename'] = filename
		self.input_params['encoding'] = encoding
		self.input_params['chunksize'] = chunksize
		self.input_params['parse_dates'] = []
		self.input_params['numerical_dtypes'] = {}
		self.input_params['dtype'] = None
		
		df = pd.read_csv(
			self.input_params['filename'], 
			encoding = self.input_params['encoding'],
			nrows = 5)
		self.input_params['usecols'] = list(df.columns.values)
		print("Peek the original data (below)")
		print(df)
		
		self.parallel = parallel
		
	# columns are all removed according to del_cols, then for col in to_date_cols
	# it will try to convert those cols into dates
	def optimize(self, to_date_cols = None, del_cols = None, show_info = False):
		
		# df is truly normal_dataframe or chunk_dataframe
		modified_df = dataframe_reader(self.input_params, self.parallel).get_dataframe()
		
		if show_info is True:
			print("====== Before optimization ====== ")
			modified_df.show_data_info()	
		
		# do optimization and modified input params
		new_input_params = copy.deepcopy(self.input_params);
		
		if del_cols is not None:
			for col in del_cols:
				new_input_params['usecols'].remove(col)
		
		if to_date_cols is not None:
			for col in to_date_cols:
				if col in new_input_params['usecols']:
					new_input_params['parse_dates'].append(col)
		
		print("Optimization Phase 1 (date-parsing\column removal) complete.")
		
		# get modified_df again with different input params
		modified_df = dataframe_reader(new_input_params, self.parallel).get_dataframe()
			
		# now, dates, useless cols are done dealt with
		# 1. convert the numeric cols into more space efficient type
		# 2. convert the object cols into categories if criteria met (unique values percentage < 50%)
		data_preprocessor.optimize_numerical_compact_type(modified_df, new_input_params)
		print("Optimization Phase 2 (int\\float compact type conversion) complete.")
	
		data_preprocessor.optimize_object_category_type(modified_df, new_input_params)
		print("Optimization Phase 3 (object to category conversion) complete.")
		
		# get modified_df again with new input params
		modified_df = dataframe_reader(new_input_params, self.parallel).get_dataframe()
		
		if show_info is True:
			print("\n====== After optimization ====== ")
			modified_df.show_data_info()	
			print("\n\n")
		
		return new_input_params;
		
	@staticmethod
	def optimize_numerical_compact_type(modified_df, new_input_params):
		# collect all result
		new_numrical_dtypes = modified_df.perform_action(dataframe_ops.judge_compact_data_type)
		new_numrical_dtypes = local_util.make_list_dict(new_numrical_dtypes)
		new_input_params['numerical_dtypes'] = local_util.merge_dict(new_numrical_dtypes, method='append')
		
		data_preprocessor.finalize_compact_type(new_input_params['numerical_dtypes'])
		#print(new_input_params['numerical_dtypes'])
		
		
	@staticmethod
	def optimize_object_category_type(modified_df, new_input_params):
		# collect all result
		unique_values = modified_df.perform_action(dataframe_ops.collect_unqiue_values)
		unique_values = local_util.merge_dict(local_util.make_list_dict(unique_values), method='unique')
		
		new_input_params['dtype'] = {}
		data_preprocessor.finalize_categorized_columns(unique_values, new_input_params['dtype'], modified_df.get_info('total_rows'))
		#print(new_input_params['dtype'])
		
	@staticmethod
	def finalize_compact_type(new_numrical_dtypes):
		type_count = max([len(tps) for _,tps in new_numrical_dtypes.items()])	
		pop_cols = []
		
		for col, tps in new_numrical_dtypes.items():
			
			if len(tps) != type_count:
				# remove from the dict
				pop_cols.append(col)
			else:
				most_compact_type = None
				for tp in tps:
					if (most_compact_type is None) or (np.dtype(tp).itemsize > np.dtype(most_compact_type).itemsize):
						most_compact_type = tp
				new_numrical_dtypes[col] = most_compact_type
		
		# for reading csv in chunk
		# the only possiblity that len(tps) != type_count is
		# when the column is sometimes not chosed as an float type
		# hence, cannot convert this column into numerical type
		print('\nColumns not consistently selected as float type:')
		print(pop_cols)
		for col in pop_cols:
			new_numrical_dtypes.pop(col)
		
		print('New numerical data types: ')
		print(new_numrical_dtypes)
	
	@staticmethod
	def finalize_categorized_columns(unique_values, new_dtypes, total_rows):
		for col, uv in unique_values.items():
			if len(uv) / total_rows < 0.5:
				new_dtypes[col] = 'category'
				
		print('\nColumns to be transformed into categories: ')		
		print([col for col,_ in new_dtypes.items()])		
	
		