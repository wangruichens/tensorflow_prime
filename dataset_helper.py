# gen prime dataset for tensorflow
import csv
import tensorflow as tf
import time
import numpy as np
from tqdm import tqdm

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed

@timeit
def generator(target):
    n = 1
    with open(target, 'w') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        for row in tqdm(csv.reader(open('prime_1b.txt')),total=10**9):
            p = int(row[0])
            while n < p - 1:
                n += 1
                if np.random.uniform(0, 1) > 0.8 and n % 2 != 0:
                    w.writerow([n, 0])
            if n == p-1:
                n += 1
                w.writerow([p, 1])


@timeit
def generator_with_even(target):
    n = 1
    with open(target, 'w') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        for row in tqdm(csv.reader(open('prime_1b.txt')),total=10**9):
            p = int(row[0])
            while n < p - 1:
                n += 1
                if np.random.uniform(0, 1) > 0.8:
                    w.writerow([n, 0])
            if n == p-1:
                n += 1
                w.writerow([p, 1])

if __name__ == '__main__':
    # cost 12 min
    generator('dataset.csv')
    # generator_with_even('dataset_even.csv')