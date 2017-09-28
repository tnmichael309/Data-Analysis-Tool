from dataframe_property_describe import dataframe_optimizer
import timeit

del_cols = ['MiddleName', 'Suffix', 'Institution', ]
to_date_cols = ['ExhibitionBeginDate', 'ExhibitionEndDate']


use_times = {
	'normal test' : 0,
	'parallel normal test' : 0,
	'chunk test' : 0,
	'parallel chunk test' : 0
    }

def normal_test():
	d = dataframe_optimizer(filename='moma.csv', encoding='utf-8', del_cols = del_cols, parse_dates = to_date_cols)
	d.optimize(show_info = True)			

def parallel_normal_test():
	d = dataframe_optimizer(filename='moma.csv', encoding='utf-8', del_cols = del_cols, parse_dates = to_date_cols)
	d.optimize(show_info = True, parallel = True)
	
def chunk_test():		
	d = dataframe_optimizer(filename='moma.csv', encoding='utf-8', del_cols = del_cols, parse_dates = to_date_cols)
	d.optimize(show_info = True, chunksize = 5000)			

def parallel_chunk_test():
	d = dataframe_optimizer(filename='moma.csv', encoding='utf-8', del_cols = del_cols, parse_dates = to_date_cols)
	d.optimize(show_info = True, parallel = True, chunksize = 5000)


if __name__ == '__main__':
	use_times['normal test'] = timeit.timeit('normal_test()', number=1, globals=globals())
	use_times['parallel normal test'] = timeit.timeit('parallel_normal_test()', number=1, globals=globals())
	use_times['chunk test'] = timeit.timeit('chunk_test()', number=1, globals=globals())
	use_times['parallel chunk test'] = timeit.timeit('parallel_chunk_test()', number=1, globals=globals())

	for k,v in use_times.items():
		print(k, '\n', v, '\n')