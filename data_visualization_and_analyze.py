import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from dataframe_property_describe import dataframe_reader

'''
	A data analysis and visualization helper
	Note that null values will be dropped
'''
class data_analyze_helper:
	default_print_space = 15
	
	def __init__(self, input_params, print_space = default_print_space):
		self.reset_params(input_params, print_space = print_space)
		
	def reset_params(self, input_params, print_space = default_print_space):
		self.input_params = input_params
		self.print_space = print_space
		self.df = dataframe_reader(input_params, False).get_dataframe()
		
		# clean data with only category and numerical types
		self.df = self.df.dropna(axis=0, how='any')
		
		del_cols = []
		for col in self.df.columns:
			type_name = self.df[col].dtype.name
			if type_name == 'object' or type_name == 'datetime64[ns]':
				del_cols.append(col)
			if type_name == 'category':
				self.df[col] = self.df[col].cat.codes
		
		for col in del_cols:
			self.df.drop(col, axis=1, inplace=True)
			
	def compare_all_to_target(self, target_col, show_graph = False):
		self.compare_cols(list(self.df.columns), [target_col], show_graph=show_graph)
		
		
	def compare_cols(self, y_cols, x_cols, hue=None, show_graph = False):
		
		print('================')
		print(' pearsonr table ')
		print('================')
		
		new_y_cols = [y for y in y_cols if y in self.df.columns]
		new_x_cols = [x for x in x_cols if x in self.df.columns]
		
		self.beautiful_print('')
		for x in new_x_cols:
			self.beautiful_print(x)
		self.beautiful_print('', change_line=True)	
			
		for y in new_y_cols:
			self.beautiful_print(y)
			for x in new_x_cols:
				# print r value only
				# print(self.df[x].dtype, self.df[y].dtype)	
				rvalue = stats.pearsonr(self.df[x], self.df[y])[0]
				rvalue = "{:.3f}".format(rvalue)
				self.beautiful_print(rvalue)
			self.beautiful_print('', change_line=True)
			
		if show_graph is True:
			g = sns.PairGrid(self.df, 
				x_vars=new_x_cols, 
				y_vars=new_y_cols,
				hue=hue, palette=sns.color_palette("husl", 8),
				size = 5)
			g.map(plt.scatter)
			g.add_legend()
			plt.show()
	
	def beautiful_print(self, s, change_line=False):
		str_len = len(s)
		if str_len > self.print_space:
			raise ValueError('String length ({}) should be less than {}'.format(str_len, self.print_space))
		
		print(s, ' '*(self.print_space - str_len), end='')
		
		if change_line is True:
			print('')
		
		