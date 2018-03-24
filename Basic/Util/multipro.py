import os
import time

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from Basic.IO import obj2file

DEBUG = True

MULTIPROCESS_FAILURE_FILE = lambda flag: 'D:/%s.txt' % flag


def put_failure_path(func):
    global MULTIPROCESS_FAILURE_FILE
    MULTIPROCESS_FAILURE_FILE = func


def loop(func, para_list, num_process = 1, flag = '', times = 1, delay = 0,
         show_seq = True, limit = -1, if_debug = False):
    if_debug = if_debug if if_debug else DEBUG

    def _process(arr):
        pid = os.getpid()
        fails = []
        result = []
        count = 0
        for para in arr:
            if show_seq:
                if isinstance(para, str):
                    print(pid, count, flag, para)
                else:
                    print(pid, count, flag, len(para))
                count += 1

            def get_val(i):
                if if_debug:
                    return func(para)
                else:
                    try:
                        return func(para)
                    except TimeoutError as e:
                        if i == times - 1:
                            print(flag, para, 'multiprocess timeout:', e)
                            fails.append([flag, para])
                        else:
                            if delay > 0:  time.sleep(delay)
                            return get_val(i + 1)
                    except Exception as e:
                        print(flag, para, 'multiprocess failures:', e)
                        fails.append([flag, para])

            res = get_val(0)
            if res is not None:
                result.append(res)
        return result, fails

    if limit > 0:
        para_list = para_list[0:limit]
    if num_process <= 1:
        final_result, failures = _process(para_list)
    else:
        arr_list = np.array_split(para_list, num_process)
        print('process num:', num_process, 'job queue length:', len(para_list))
        with Pool() as pool:
            outs = pool.map(_process, arr_list)
            # pool.close()
            # pool.join()
        final_result = [r for x in outs for r in x[0]]
        failures = [f for x in outs for f in x[1]]
    if len(failures) > 0:
        obj2file(MULTIPROCESS_FAILURE_FILE(flag), failures)
    return final_result
