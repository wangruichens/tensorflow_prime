# tensorflow_prime
Using tensorflow deep nn to determine whether a number is prime...

[素数](https://github.com/wangruichens/notes/blob/master/prime%20number/prime.pdf)

##

虽然这是一个不可能的任务，但一样可以学到一些比较有意思的东西。

* 素数生成算法 埃氏筛除法
* 数据转tfrecord为tf.data.Dataset服务，当然也可以直接用csv输入,为了使用和适配更高阶的API
* tf.Example 作为tfrecord的{"string": tf.train.Feature} mapping方法，序列化得很慢
* tfrecord的反序列化
* tensorflow+keras 模型train,fit,eval,predict
* keras超级方便的多gpu并行训练
* tensorboard 查看模型结构，loss,acc曲线

##

gen_prime.py 计算10亿（10 **9）内的素数,大概需要40G的内存

dataset_helper.py 等比例的在10亿内素数中混入合数，标记素数label为1，合数为0。 存为pickle格式，数据大概 1.8G。

tfrecord.py 将csv数据集生成tfrecord

tensorflow_isprime_tiny_example.py 100万的素数DNN小例子，输入为二进制数字

tensorflow_isprime.py 10亿的素数DNN模型，30位二进制表示。10亿内素数+合数

##
素数相关猜想：

哥德巴赫猜想，黎曼猜想，孪生素数猜想