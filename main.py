from timeit import timeit

from a_test import test

if __name__ == '__main__':
    print('start')
    print(timeit(test, number=1))
    times = 100
