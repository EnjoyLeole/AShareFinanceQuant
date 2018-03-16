import time
import os
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool


def loop(func, para_list, num_process = 1, flag = '', times = 1, delay = 0,
         show_seq = False, limit = -1):
    def _process(arr):
        pid = os.getpid()
        fails = []
        count = 0
        for para in arr:
            if show_seq:
                print(pid, count, flag, para)
                count += 1
            for i in range(times):
                try:
                    func(para)
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

        return fails

    if limit > 0:
        para_list = para_list[0:limit]
    if num_process <= 1:
        failures = _process(para_list)
    else:
        arr_list = np.array_split(para_list, num_process)
        print(num_process, len(para_list))
        pool = Pool()
        fails = pool.map(_process, arr_list)
        pool.close()
        failures = []
        for f in fails:
            failures += f
    return failures
