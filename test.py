from data_preprocess import data_preprocessor
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
	d = data_preprocessor('moma.csv')
	d.optimize(to_date_cols=to_date_cols, del_cols=del_cols, show_info=True)		

def parallel_normal_test():
	d = data_preprocessor('moma.csv', parallel = True)
	d.optimize(to_date_cols=to_date_cols, del_cols=del_cols, show_info=True)
	
def chunk_test():		
	d_chunk = data_preprocessor('moma.csv', chunksize = 10000)
	d_chunk.optimize(to_date_cols=to_date_cols, del_cols=del_cols, show_info=True)			

def chunk_reuse_test():
	d_chunk = data_preprocessor('moma.csv', chunksize = 10000)
	d_chunk.optimize(to_date_cols=to_date_cols, del_cols=del_cols, show_info=True)
	d_chunk.optimize(to_date_cols=to_date_cols, del_cols=del_cols, show_info=True)	

def parallel_chunk_test():
	d_chunk = data_preprocessor('moma.csv', chunksize = 10000, parallel=True)
	d_chunk.optimize(to_date_cols=to_date_cols, del_cols=del_cols, show_info=True)	


if __name__ == '__main__':
	#use_times['normal test'] = timeit.timeit('normal_test()', number=1, globals=globals())
	use_times['parallel normal test'] = timeit.timeit('parallel_normal_test()', number=1, globals=globals())
	#use_times['chunk test'] = timeit.timeit('chunk_test()', number=1, globals=globals())
	#use_times['parallel chunk test'] = timeit.timeit('parallel_chunk_test()', number=1, globals=globals())
	#chunk_reuse_test()	

	for k,v in use_times.items():
		print(k, '\n', v, '\n')