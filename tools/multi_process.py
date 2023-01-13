from multiprocessing import Pool
class multi_process():
    def __init__(self, num_core, file_list):
        self.num_core = num_core
        len_file = len(file_list)
        self.List_subsets = []
        for i in range(num_core):
            if i != num_core - 1:
                subset = file_list[(len_file * i) // num_core:(len_file * (i+1)) // num_core]
            else:
                subset = file_list[(len_file * i) // num_core:]
            self.List_subsets.append(subset)
    def apply_async(self, single_worker):
        p = Pool(self.num_core)
        for i in range(self.num_core):
            p.apply_async(single_worker, args=(self.List_subsets[i], ))
        p.close()
        p.join()