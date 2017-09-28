
from dataframe_property_describe import dataframe_optimizer, dataframe_reader
from data_visualization_and_analyze import data_analyze_helper

if __name__ == '__main__':
	
	del_cols = ['MiddleName', 'Suffix', 'Institution','ExhibitionID', 'ExhibitionTitle', 'ExhibitionCitationDate']
	to_date_cols = ['ExhibitionBeginDate', 'ExhibitionEndDate']

	# get an optimizer
	d = dataframe_optimizer(filename='moma.csv', encoding='utf-8', del_cols = del_cols, parse_dates = to_date_cols)
	
	# get the optimized input param of the dataframe 
	info = d.optimize(show_info = True)
	
	'''
		<optional> info can be dumped for future use without parse the data for optimization again
		dataframe_optimizer.dump_dataframe_params(info, 'test_info.dmp')
	'''
	
	'''
		<optional> 
			1. initialized the dataframe_reader with that info and get the dataframe (optimized)!
			2. do data preprocess
			3. write to csv
			4. repeat dataframe optimize steps above to get optimized info
		df = dataframe_reader(info).get_dataframe()
		#do data preprocess: 
		#	deal with missing values (too many-> discard col, ok->fill by average or discard that row), 
		#   merge cols
		df.write_to_csv(new_csv) or dump
		d = dataframe_optimizer(new_csv)...
		info = d.optimize()
		dataframe_optimizer.dump_dataframe_params(info, 'test_info.dmp')
	'''
	
	# initialize a data analyze and visualization helper
	# which internally maintained a dataframe created with provided input_params
	data_analyzer = data_analyze_helper(input_params=info, print_space=30)
	#data_analyzer.compare_all_to_target(target_col='Gender', show_graph = True)
	data_analyzer.compare_cols(['AlphaSort','ExhibitionNumber'], ['Nationality'], hue='Gender', show_graph = True)