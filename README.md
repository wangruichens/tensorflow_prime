# tensorflow_prime
Using tensorflow deep nn to determine whether a number is prime :P

[素数](https://github.com/wangruichens/notes/blob/master/prime%20number/prime.pdf)


gen_prime.py 计算10亿（10 **9）内的素数

dataset_helper.py 等比例的在10亿内素数中混入合数，标记素数label为1，合数为0

tensorflow_isprime_tiny_example.py 100万的素数DNN小例子，输入为二进制数字

tensorflow_isprime_tiny_example.py 10亿的素数DNN模型，30位二进制表示。10亿内素数+合数，csv数据大概 1.5G