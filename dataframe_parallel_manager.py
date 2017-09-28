import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing

class df_parall_manager:
	def __init__(self):
		pass
		
	@staticmethod
	def get_cpu_count():
		return multiprocessing.cpu_count() - 1
		
	# dataframe (df) apply some function (func) with variant arguments (*args)
	# return a new df
	@staticmethod
	def apply_parallel_df_action(df, func, *args):
	
		cpu_count = df_parall_manager.get_cpu_count
		df_split = np.array_split(df, cpu_count)
		
		with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
			packed_args = ((func, chunk, *args) for chunk in df_split)
			res = executor.map(unpack_args_and_execute, packed_args)
			res = list(res)
			return pd.concat(res)
		
		return None
	
	@staticmethod
	def unpack_args_and_execute(packed_args):
		if len(packed_args) > 2:
			return packed_args[0](packed_args[1], packed_input[1])
		else:
			return packed_args[0](packed_args[1])
	
	