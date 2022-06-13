import sys
import re_main

dataset = sys.argv[1]
res = sys.argv[2]
gpu_id = sys.argv[3]
    
re_main.DATASET = dataset
re_main.RES = res
re_main.GPU_ID = gpu_id


training = {
    'window_width' : 10,
    'kernel_num' : 20,
    'epochs' : 10,
    'batch_size':  32, 
    'learning_rate' : 0.1,
    'target_from_theo_pept' : 0.005,
    'topN': 5000,
    'clip_value': 1,
    'cand_from_search_qval' : 0.005,
    'print_info': True,
    'printing_tick': 1 # print learning update after this many epochs
}

filename_stem = main_train.main(training)
