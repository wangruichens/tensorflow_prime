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


def to_bin(x,y, bins=30):
    binary = bin(x)[2:].zfill(bins)
    binary +=str(y)
    return [int(b) for b in binary]

# 50847534 prime number less than 10**9
@timeit
def generator(target):
    n = 1
    pos=0
    neg=0
    with open(target, 'w') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        for row in tqdm(csv.reader(open('prime_1b.txt')),total=50847534):
            p = int(row[0])
            while n < p - 1:
                n += 1
                if np.random.uniform(0, 1) > 0.94:
                    w.writerow(to_bin(n,0))
                    neg+=1
            if n == p-1:
                n += 1
                w.writerow(to_bin(n,1))
                pos+=1
    print('done, pos num={0},neg={1}'.format(pos,neg))

if __name__ == '__main__':
    # cost 12 min
    generator('data/dataset_1b.csv')