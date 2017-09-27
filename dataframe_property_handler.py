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
	