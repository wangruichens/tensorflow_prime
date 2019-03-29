import  time

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


def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n

def _not_divisible(n):
    return  lambda x:x%n>0

def primes():
    yield 2
    it=_odd_iter()
    while True:
        n=next(it)
        yield n
        it=filter(_not_divisible(n),it)

def to_bin(x,bins=20):
    str=bin(x)[2:].zfill(bins)
    return [int(b) for b in str]

# Eratosthenes algorithm
# O(N*loglogN)
@timeit
def list_primes(n):
    if n < 3:
        return 0
    primes = [True] * n
    primes[0] = primes[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if primes[i]:
            primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
    return primes

# Euler algorithm
# every composite number will be killed by its minimal prime number
@timeit
def euler(lens):
    ss = []
    check = [True]*(lens + 1)
    for i in range(2, lens):
        if check[i]:
            ss.append(i)
        for j in ss:
            if j * i > lens:
                break
            check[j * i] = False
            if i % j == 0:
                break
    return ss

# start_time = time.time()
# lens=10**1
# with open('prime_iter.txt','w') as f:
#     for n in primes():
#         if n<lens:
#            # write to file
#             f.write(str(n)+ '\n')
#         else:
#             break
#     f.close()
# elapse_slow=time.time() - start_time
# print('finished... time: ',elapse_slow)

start_time = time.time()
lens=10**6
l=list_primes(lens)
# l2=euler(lens)
with open('prime_1m.txt','w') as f:
    for n in range(lens):
        if l[n]==True:
           # write to file
            f.write(str(n)+ '\n')
    f.close()
elapse_fast=time.time() - start_time
# 1 billion prime , cost 420s (7min)
print('finished ... time: ',elapse_fast)