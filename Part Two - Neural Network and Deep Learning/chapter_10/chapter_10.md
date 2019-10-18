# Chapter 10: Introduction to Artificial Neural Network

This chapter introduces how to create an Aritificial Neural Network using TensorFlow, both with higher level APIs and lower level operators. TensorFlow provides adequate flexibility for users to create a deep learning model. It will be much easier and faster to create a model using Keras, but with TensorFlow you can do more things. For more details on TensorFlow, check out the [official website](https://tensorflow.google.cn).

>   This markdown file contains my answers to the questions of the book. For coding exercises please refer to the jupyter notebook under the same directory.

2.  为什么通常更倾向用逻辑回归分类而不是经典的感知器？如何调整一个感知器，让它与逻辑回归分类器等价？

    经典的感知器无法处理非线性问题。

    为感知器添加 sigmoid 激活函数即可使其与逻辑回归分类器等价。

    >   **Answer**
    >
    >   经典的感知器只有在数据集是线性可分的情况下才会收敛，并且不能估计分类的概率。作为对比，逻辑回归分类器即使在数据集不是线性可分的情况下也可以很好地收敛，而且还能输出分类的概率。如果将感知器的激活函数改为逻辑激活函数，然后训练其使用梯度下降，就会变成逻辑回归分类器了。

3.  为什么逻辑激活函数是训练第一个 MLP 的关键因素？

    逻辑激活函数为感知器引入了非线性，使 MLP 能够解决非线性问题，并且可以通过反向传播对梯度进行更新。

    >     **Answer**
    >
    >     逻辑激活函数的导数总是非零的，所以梯度下降总是可以持续的。当激活函数是阶跃函数时，渐变下降就不能再持续了，因为这时候没有斜率。

4.  说出 3 种流行的激活函数。

    1.  ReLU
$$
f(x)=\left\{
\begin{array}{rl}
0 \quad x < 0 \\
x \quad x \ge 0 \\
\end{array} \right.
$$
    2.  sigmoid
$$
f(x) = \frac{1}{1+e^{-x}}
$$
    3.  tanh
$$
f(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}
$$

5.  假设你有一个 MLP 包含：由一个有 10 个透传神经元组成的输入层，及一个有 50 个人工神经元的隐藏层，以及一个有 3 个神经元的输出层。所有的神经元都用 ReLU 激活函数。那么：

    -   输入矩阵 X 的形状是什么？
    -   隐藏层权重向量 $W_h$，偏移向量 $b_h$ 的形状呢？
    -   输出层权重向量 $W_o$，偏移向量 $b_o$ 的形状呢？
    -   输出矩阵 Y 的形状是什么？
    -   写出计算网络输出矩阵 Y 对应 X、$W_h$、$b_h$、$W_o$ 和 $b_o$ 的方程式。

    输入矩阵 X 的形状是 $ m \times 10 $，其中 m 为样本数量。

    $ W_h $ 的形状为 $ 10 \times 50 $，$ b_h $ 的形状为 50。
    
    $ W_o $ 的形状为 $ 50 \times 3 $，$ b_o $ 的形状为 3。
    
    输出矩阵 Y 的形状是 $ m \times 3$。
    
    方程式为
    $$
    Y = f(f(XW_h + b_h)W_o + b_o)
    $$
    
    其中 ```f(x)``` 为 ReLU 激活函数。

6.  要区分邮件是不是垃圾邮件，输出层需要多少神经元？输出层应该选择哪种激活函数？如果要处理 MNIST，输出层又需要多少个神经元？使用哪种激活函数？

    区分垃圾邮件，输出层需要 1 个神经元即可，可以采用 sigmoid 激活函数。

    处理 MNIST，输出层需要 10 个神经元，可以采用 softmax 激活函数。

7.  什么是反向传播，它是如何工作的？反向传播与反式自动微分有何区别？

    反向传播首先计算网络输出与标签之间的误差，然后根据误差，对各层的权重进行更新。

    >     **Answer**
    >
    >     反向传播首先计算关于每个模型参数的成本函数的梯度，然后使用这些梯度执行梯度下降。为了计算梯度，反向传播使用反向模式 autodiff，会现在计算图上正向执行一次，计算当前训练批次的每个节点的值，然后反向执行一次，一次性计算所有梯度。
    >
    >     作为对比，反向传播是指使用多个反向传播步骤来训练人工神经网络的全部过程，每个步骤计算梯度并使用它们执行梯度下降过程。而反向模式只是一种简单的计算梯度的技术，只是恰好被反向传播使用了而已。

8.  你能列出可以被调整的所有的 MLP 的超参数吗？如果 MLP 对于数据集过度拟合了，你会如何调整这些超参数来解决？

    可以调整的超参数包括：隐藏层的个数、隐藏层包含的神经元个数、训练批次数量、每批训练样本的数量、激活函数的选择、优化器的类型、学习率等。

    >     **Answer**
    >
    >     隐藏层的数量、每个隐藏层中神经元的数量，以及每个隐藏层和输出层中使用的激活函数。一般来说，ReLU 激活函数时隐藏层的一个很好的默认值。对于输出层，通常需要二分类的逻辑激活函数，多分类的 softmax 激活函数，在做回归时则无需激活函数。
    >
    >     如果 MLP 对训练数据有过度拟合，可以尝试减少隐藏层的数量，并减少每个隐藏层的神经元数量，使模型的复杂度下降，缓解过拟合的现象。