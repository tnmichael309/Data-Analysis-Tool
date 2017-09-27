
class local_util:
	@staticmethod
	def merge_dict(list_of_dicts, method = 'add'):
		ret_dict = {}
		for dict in list_of_dicts:
			for k, v in dict.items():
				if method == 'add':
					if k not in ret_dict:
						ret_dict[k] = 0
					ret_dict[k] += v
				elif method == 'append':
					if k not in ret_dict:
						ret_dict[k] = []
					ret_dict[k].append(v)
				elif method == 'extend':
					if k not in ret_dict:
						ret_dict[k] = []
					ret_dict[k].extend(v)
				elif method == 'unique':
					if k not in ret_dict:
						ret_dict[k] = []
					for val in v:
						if val not in ret_dict[k]:
							ret_dict[k].append(val)	
				else:
					raise ValueError("method can be None, \"add\", \"append\", \"extend\", or \'unique\'")
					
		return ret_dict
		
	@staticmethod
	def make_list_dict(potential_dict_list):
		if type(potential_dict_list) == type({}):
			return [potential_dict_list]
		else:
			return list(potential_dict_list)