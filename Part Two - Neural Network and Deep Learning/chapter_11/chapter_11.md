# Chapter 11: Training Deep Neural Network

This chapter introduces some common techniques to train a deep neural network, including selecting initializers, choosing activation functions, batch normalization, using better optimizers, etc. This book only contains basic concept and usage of these techniques. For other techniques in training NN or the priciples of these methods, please refer to other materials.

>   This markdown file contains only my answers and solutions to the questions of chapter 11. For coding exercises, please refer to the jupyter notebook under the same directory.

1.  只要使用 He 初始化随机进行选择，就可以将所有权重初始化为相同的值吗？

    不能，随机选择是需要令各个权重的值不同，增加初始化的随机性，使得神经网络的训练能够收敛。

    >**Answer**:不能，所有权重需要独立处理，不可以初始化为统一值。随机取样权重的一个重要目的是**破坏对称性**：如果所有的权重初始化为相同的值，即使该值不为 0，对称性也无法被破坏（即一层中的所有神经元都是一样的），并且反向传播将无法破坏它。具体来说，这就意味着一层中所有神经元始终保持相同的权重，就像每层只有一个神经元，而且要慢得多。这样的配置是无法收敛到一个好的解决方案的。

2.  将偏移项初始化为 0 可以吗？

    可以，也可以采用随机初始化的方式对偏移项进行处理。

    >**Answer**:可以设置为 0，也可以随机初始化，两者没有太大区别。

3.  给出 ELU 相比 ReLU 的 3 个优点。

    -   在 0 处的导数有定义，提高了梯度下降的稳定性。
    -   在输入为负值的时候 ELU 的值不为零，避免了神经元“死亡”的问题。
    -   ELU 比 ReLu 更加平滑。

    >**Answer**: ELU 相对 ReLU 的几个优势：
    >
    >-   它可以使用负值，所以相比使用 ReLU，某一给定层的神经元输出平均值理论上更容易接近 0，这样有助于缓解梯度消失问题。
    >-   它总是有一个非零的导数，可以避免影响 ReLU 单元的单元消失问题。
    >-   它在任何地方都是平滑的，而在 0 处，ReLU 的梯度突然从 0 跳至 1，这个突然的变化会引起在 0 附近摆动，从而可以缓解梯度下降。

4.  在什么情况下你会依次使用下列这些激活函数：ELU、Leaky ReLU（以及它的变体）、ReLU、tanh、逻辑激活函数和 softmax？

    -   在大多数情况下都可以使用 ELU。
    -   如果需要较快的收敛速度，可以使用 Leaky ReLu 或 ReLU。
    -   对于正负样本的分类任务可以采用 tanh或者逻辑激活函数
    -   如果需要给出预测类别的概率值，可以采用逻辑激活函数
    -   在训练样本有多种标签的输出层使用 softmax。

    >**Answer**: ELU 激活函数是一个不错的默认选择。如果对神经网络的速度要求很高，可以用 Leaky ReLU 的一个变种（即使用默认超参数值的 Leaky ReLU）。这是因为 ReLU 激活函数简单方便，所以很多人会将其作为首选，即使输出表现会被 ELU 和 Leaky ReLU 超过。但是，在某些情况下，ReLU 激活函数的精确输出能力是有用的。如果你需要输出一个介于 -1 和 1 之间的数，tanh 在输出层会比较有效，但是现在在隐藏层的使用频度并不高。在你需要评估可能性时，逻辑激活函数在输出层比较有效，但是同样在隐藏层中很少使用。最后 softmax 激活函数在输出层输出互相排斥类的概率是有效的，但是除了隐藏层以外基本不用。

5.  使用 Momentum Optimizer 时，如果你将动量超参数设置得离 1 特别近，那么会发生什么？

    动量值过大可能会导致算法产生震荡，使收敛速度下降

    >**Answer**: 算法会提速很高，偏向全局最小值，但是接着会经过最小值。之后就会慢慢降速回落，再加速，再超调，循环往复。这种方式在收敛前会震荡好几次，所以总体来说，收敛速度会比用小动量慢。

6.  给出三种沟通稀疏模型的方法。

    -   使用 L1 正则化对权重进行筛选
    -   

    >**Answer**: 一种实现稀疏模型的方法（即大多数权重等于 0 ）是正常训练一个模型，然后将小权重设置为 0。为了更稀疏，可以在训练过程中使用 L1 正则化，这样可以促使优化器更加稀疏。第三种方式是使用 TensorFlow 的 FTRLOptimizer 类将 L1 正则化和对偶平均相结合。

7.  Dropout 会减慢训练速度吗？是否减慢推理？

    Dropout 会减慢训练速度，因为需要随机丢弃一部分神经元，但是在推理时没有影响。

    >**Answer**: Dropout 会减慢训练速度，一般降为原速度的一半。但是，因为只是在训练期间使用，所以对于预测没有影响。