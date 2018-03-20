import time
import os
import numpy as np
from Basic.IO import obj2file
from pathos.multiprocessing import ProcessingPool as Pool

DEBUG = True

MULTIPROCESS_FAILURE_FILE = lambda flag: 'D:/%s.txt' % flag


def set_failure_path(func):
    global MULTIPROCESS_FAILURE_FILE
    MULTIPROCESS_FAILURE_FILE = func


def loop(func, para_list, num_process = 1, flag = '', times = 1, delay = 0,
         show_seq = False, limit = -1, if_debug = False):
    if_debug = if_debug if if_debug else DEBUG

    def _process(arr):
        pid = os.getpid()
        fails = []
        result = []
        count = 0
        for para in arr:
            if show_seq:
                print(pid, count, flag, para)
                count += 1
            for i in range(times):
                res = None
                if if_debug:
                    res = func(para)
                    break
                else:
                    try:
                        res = func(para)
                        break
                    except TimeoutError as e:
                        if i == times - 1:
                            print(e)
                            fails.append([flag, para])
                        if delay > 0:
                            time.sleep(delay)
                    except Exception as e:
                        print(e)
                        fails.append([flag, para])
                        break
                if res is not None:
                    result.append(res)

        return result, fails

    if limit > 0:
        para_list = para_list[0:limit]
    if num_process <= 1:
        final_result, failures = _process(para_list)
    else:
        arr_list = np.array_split(para_list, num_process)
        print(num_process, len(para_list))
        pool = Pool()
        results, fails = pool.map(_process, arr_list)
        pool.close()
        failures = []
        for f in fails:
            failures += f

        final_result = []
        for r in results:
            final_result += r
    obj2file(MULTIPROCESS_FAILURE_FILE(flag), failures)
    return final_result
