# Chapter 9: Running TensorFlow

This chapter introduces how to install TensorFlow and how to use it to create a computing graph. The power of TensorFlow is that it can compute gradient automatically. For further information and tutorials on how to use TensorFlow, please refer to the documentation or other TensorFlow tutorials.

> This markdown file contains my answers to the questions below the chapter. For coding exercise, please check the jupyter notebook under the same folder.

1. 相比直接执行计算，创建计算图的最大优点是什么？最大的缺点呢？

优点：创建计算图可以将计算的定义和执行分离开来，便于进行并行计算和分布式计算。

缺点：消耗比较多的运算资源。

> **Answer:**
>
> 主要优点：
>
> - 可以自动计算梯度
> - 负责在不同的线程中并行执行各个操作
> - 可以更容易地在多设备上运行同一个模型
> - 简化了查看，可以使用 TensorBoard 进行可视化
>
> 主要缺点：
>
> - 学习曲线陡峭
> - 逐步的调试比较困难

2. 语句 ```a_val = a.eval(session=sess)``` 和 ```a_val= sess.run(a)``` 等价吗？

两者等价，后一种方式需要手动关闭会话。

> **Answer:**
>
> 两者完全等价。

3. 语句 ```a_val, b_val = a.eval(session=sess), b.eval(session=sess)``` 和 ```a_val, b_val = sess.run([a, b])``` 等价吗？

前一种方式对 a、b 两个操作进行串行计算，后一种方式同时对两个操作进行计算。

> **Answer:**
>
> 前一种方式会计算两次；后一种方式只计算一次，并且会快一点。如果这些操作中的任意一个具有副作用，效果就会不同。

4. 你可以在同一个会话中运行两个图吗？

可以创建不同命名空间内的两个图，但是一次只能够运行一个。

> **Answer:**
>
> 无法在同一个会话中运行两个计算图，需要将两个图合并为一个大图。

5. 假设你创建了一个包含变量 w 的图，然后在两个线程中分别启动一个会话，两个线程都使用了图 g，每个会话会有自己对 w 变量的拷贝，还是会共享变量？

每个会话拥有对 w 变量的拷贝。

> **Answer:**
>
> 在本地 TensorFlow 中，会话用来管理变量的值，每个会话拥有自己对变量的拷贝。如果在分布式 TensorFlow 中，变量的值则会存储在由集群管理的容器中，如果两个会话连接了同一个集群，并使用同一个容器，那么它们会共享变量。

6. 变量何时被初始化，又在何时被销毁？

在执行```initializer```的时候创建变量，在会话结束的时候销毁变量。

> **Answer:**
>
> 变量在调用其初始化器的时候被初始化，在会话结束的时候被销毁。在分布式 TensorFlow 中，变量存活于集群上的容器中，所以关闭一个会话不会销毁变量，只有清空变量所在的容器才能够销毁变量。

7. 占位符和变量的区别是什么？

占位符只需要确定输入的维度，只要满足维度要求，在执行的时候通过```feed_dict```传入数据。

变量需要有初始化的步骤，即变量一开始就会被赋予一个确定的值。

> **Answer:**
>
> 两者完全不同。
>
> - 变量是包含一个值的操作。你执行一个变量，它会返回对应的值。在执行之前，需要初始化变量。可以修改变量的值。变量有状态：在连续运行图时，变量保持相同的值。通常它被用作保存模型的参数。
> - 占位符只有其所代表的张量的类型和形状的信息，但没有值。如果要对一个依赖于占位符的操作进行求值，必须先通过 feed_dict 对其进行传值，否则会得到一个异常。占位符通常被用作在执行期为训练或者测试数据传值。

8. 如果对一个依赖于占位符的操作求值，但是又没有为其传值，会发生什么？如果这个操作不依赖于占位符呢？

会引发异常，必须为占位符传值才能使程序正常运行。

如果操作不依赖于占位符，程序能够正常运行。

> **Answer:**
>
> 如果运行计算图来求值一个依赖于占位符的操作，但不提供值，则会发生异常。如果操作不依赖于占位符，则不会引发异常。

9.  运行一个图时，可以为任意操作输出值，还是只能输出占位符的值？

可以为任意操作输出值。

> **Answer:**
>
> 在运行一个图时，可以提供任何操作的输出值。

10. 在执行期，你如何为一个变量设置任意的值？

不能在执行期为变量赋值。

> **Answer:**
>
> 最简单的方式是使用```tf.assign```创建一个赋值节点，将变量和一个占位符传入作为参数，这样就可以在执行期运行赋值操作来为变量传入新值。
>
> 示例代码如下：
>
> ``` python
> x = tf.Variable(tf.random_uniform(shape=(), 0.0, 1.0))
> x_new_val = tf.placeholder(shape=(), dtype=tf.float32)
> x_assign = tf.assign(x, x_new_val)
> 
> with tf.Session() as sess:
>     x.initializer.run()
>     print(x.eval())
>     x_assign.eval(feed_dict={x_new_val: 5.0})
>     print(x.eval())
> ```

11. 反向模式 autodiff 需要多少次遍历图形才能计算 10 个变量的成本函数的梯度？正向模式 autodiff 怎么样？符号微分呢？

No idea :(

> **Answer:**
>
> 要计算任意数量变量的成本函数的梯度，反向模式只需要遍历两次图，正向模式需要为每个变量运行一次。对于符号微分，它会建立一个不同的图来计算梯度，不会遍历原来的图，但是新的图可能是非常复杂和低效的。
