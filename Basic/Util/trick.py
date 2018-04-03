import random
from uuid import uuid4


def print_num(*arg):
    line = ''
    for ar in arg:
        if isinstance(ar, float) or isinstance(ar, int):
            line += '%e  ' % ar
        else:
            line += '%s  ' % ar

    print(line)


def uid():
    unique_id = uuid4()
    return str(unique_id)


def split(number, n_shares):
    if n_shares == 0:
        return number
    avg = int(number / n_shares)
    surplus = int(number - avg * n_shares)
    result = [avg] * n_shares
    for sur in range(surplus):
        result[sur] += 1
    return result


def max_at(array, evaluate_func, limit_func=None, show=False):
    peak_value = None
    chosen_list = []

    for element in array:
        if (limit_func is not None and limit_func(element)) or limit_func is None:
            cur_value = evaluate_func(element)
            if show:
                print(element, cur_value)
            if peak_value is None:
                peak_value = cur_value
                chosen_list.append(element)
            elif cur_value > peak_value:
                peak_value = cur_value
                chosen_list = [element]
            elif cur_value == peak_value:
                chosen_list.append(element)

    n = len(chosen_list)
    if n:
        chosen = chosen_list[random.randrange(n)]
        if show:
            print("result", chosen)
        return chosen
    else:
        return None


def min_at(array, evaluate_func, limit_func=None, show=False):
    return max_at(array, lambda x: -1 * evaluate_func(x), limit_func, show)


def func_name_from_stack(idx=-1):
    import traceback
    return traceback.extract_stack()[idx][2]


class Counter(object):

    def __init__(self, step=1):
        self.i = 0
        self.cur_step = 0
        self.step = step

    @property
    def portion_step(self):
        return int(self.i / self.step)

    def add_print(self):
        self.i += 1
        if self.portion_step > self.cur_step:
            self.cur_step = self.portion_step
            print(self.cur_step)
