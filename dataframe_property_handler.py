import numpy as np

'''
	df information
	
	{
		total_row :
		total_mem :
		col_info :
		{
			col_1:
			{
				dtype: current used type
				numerical_dtype: for float cols, most compact type
				is_to_cat: for obj cols, whether to transform into category type
			}
			col_2:
			...
		}
	}
'''
class dataframe_property_handler:
	def __init__(self):
		pass
		
	@staticmethod
	def print_info(dict):
		print('======================')
		print('Dataframe Infomation')
		print('======================')
		print('total_row : ', dict['total_row'])
		print('total_mem : %.2f (MB)' % (dict['total_mem']/(1024*1024)) )
		print('col_info : ')
		for col, col_dict in dict['col_info'].items():
			print('\t',col,':')
			print('\t\tdtype: ', col_dict['dtype'])
			print('\t\tnumerical_dtype (for float cols) : ', col_dict['numerical_dtype'])
			print('\t\tis_to_cat       (for object cols): ', col_dict['is_to_cat'])
			
	@staticmethod
	def get_empty_df_information_dict():
		ret = {}
		ret['total_row'] = 0
		ret['total_mem'] = 0
		ret['col_info'] = {}
		
		return ret
	
	@staticmethod
	def merge_df_information_dict(list_of_dicts):
		ret = dataframe_property_handler.get_empty_df_information_dict()
		
		for dict in list_of_dicts:
			ret['total_row'] += dict['total_row']
			ret['total_mem'] += dict['total_mem']
			
			for col, col_dict in dict['col_info'].items():
				if col not in ret['col_info']:
					ret['col_info'][col] = col_dict
				else:
					ret_dtype = ret['col_info'][col]['dtype']
					ret_numerical_dtype = ret['col_info'][col]['numerical_dtype']
					ret_unique_values = ret['col_info'][col]['is_to_cat']
		
					check_dtype = col_dict['dtype']
					check_numerical_dtype = col_dict['numerical_dtype']
					check_unique_values = col_dict['is_to_cat']
					
					if (ret_dtype == 'float64' and check_dtype == 'float64' 
						and np.dtype(check_numerical_dtype).itemsize > np.dtype(ret_numerical_dtype).itemsize):
						ret_numerical_dtype = check_numerical_dtype
					
					if (check_dtype == 'object'):
						if ret_unique_values is None:
							ret_unique_values = check_unique_values
						else:
							if type('ret_unique_values') != type([]):
								ret_unique_values = list(ret_unique_values)
							ret_unique_values.extend(list(check_unique_values))
							ret_unique_values = list(np.unique(ret_unique_values))
						
		for col, col_dict in ret['col_info'].items():
			if col_dict['is_to_cat'] is None:
				pass
			elif len(col_dict['is_to_cat']) / ret['total_row'] >= .5:
				col_dict['is_to_cat'] = False
			else:
				col_dict['is_to_cat'] = True
						
		return ret			
						
	
						
						
						
						
						
						