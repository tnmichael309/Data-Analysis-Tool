import pandas as pd

# Each ops should be design to be independent to each args (for multi-threading)
# t1: func(df1, args), t2: func(df2, args) ==> t1, t2 can only read\add items to args, not to change value\delete
# otherwise, error will be raised
'''
A class to collect all static operations applied to the dataframe
'''			
class dataframe_ops:
	def __init__(self):
		pass
	
	# Brief: collect memory_footprints, rows, null_counts of float columns into data_info (args[0])
	@staticmethod
	def collect(df, *args):
		data_info = {}
		data_info['memory_footprints'] = 0
		data_info['total_rows'] = 0
		data_info['null_counts'] = []
		
		data_info['memory_footprints'] = df.memory_usage(deep=True).sum()
		data_info['total_rows'] = len(df)
		data_info['null_counts'] = df.select_dtypes(include=['float']).isnull().sum()
	
		return data_info
		
	# Brief: use data_info (args[0]) and df to judge the compact numeric type, and return a dict
	# ret: {col: data_type_in_string}
	@staticmethod
	def judge_compact_data_type(df, *args):
		data_info = args[0]
		ret = {}
		
		float_cols = df.select_dtypes(include=['float'])
		for col in float_cols.columns:
			if data_info['null_counts'][col] != 0:
				ret[col] = pd.to_numeric(df[col], downcast='float').dtype
			else:
				ret[col] = pd.to_numeric(df[col].astype('int'), downcast='integer').dtype
	
			ret[col] = str(ret[col])
		return ret	
			
	# Brief: use data_info (args[0]) and df to collect all the unique values into args[1]
	# args[1]: {col: [array of unique values]}
	@staticmethod
	def collect_unqiue_values(df, *args):
		data_info = args[0]
		unique_values = {}
		
		obj_cols = df.select_dtypes(include=['object'])
		for col in obj_cols.columns:
			unique_values[col] = df[col].unique()
			
		return unique_values