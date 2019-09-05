# Chapter 7: Ensemble Learning and Random Forest

This chapter introduces some popular ensemble methods in machine learning, including voting, bagging, boosting and random forest. For further details on how these algorithms work, please refer to other machine learning tutorials.

> This markdown file contains only my answer to the questions of the book. For coding exercises, check the jupyter notebook under the same directory.

1. 如果你已经在完全相同的训练集上训练了五个不同的模型，并且它们都达到了 95% 的准确率，是否还有机会通过结合这些模型来获得更好的结果？

可以通过集成学习的方式进一步得到更好的结果，例如，对于一个新实例，用这五个不同的模型进行预测，并取占大多数的结果作为对该实例的预测。

> **Answer:** 可以尝试将它们组合成一个集成模型。

2. 硬投票分类器和软投票分类器有什么区别？

硬投票分类器根据多个模型对新实例的预测结果，取占多数的作为最终结果；软投票分类器要求模型能够输出概率值，对概率值做进一步的计算得到对新实例的预测概率。

> **Answer:**  硬投票分类器只是统计每个分类器的投票，然后挑选出得票最多的类别。软投票分类器计算出每个类别的平均估算概率，然后选出概率最高的实例。软投票分类器的表现更好，因为它给予那些高度自信的投票更高的权重。但是它要求每个分类器都能够估算出类别概率才可以正常工作。

3. 是否可以通过在多个服务器上并行来加速 bagging 集成的训练？pasting 集成呢？boosting 集成呢？随机森林或 stacking 集成呢？

注：在这里区分开 stacking 和 blending。stacking 指的是使用同一批训练数据训练多个模型后，利用模型输出的概率值构成新的特征，再训练次级分类器；blending 指的是将数据集划分为 k 个子集 ${D_1, D_2, \dots, D_k}$，在 $D_j$ 上训练第 j 层的模型，这也是书中介绍的 stacking。

- 可以通过在多个服务器上并行来加速的集成学习方法有：bagging, pasting, random forest
- boosting 不能并行，因为下一个模型的训练样本权重依赖于上一个模型分类错误的实例数量。
- stacking 也不能并行，因为下一层的模型训练依赖于上一层模型对留存集的预测结果。


> **Answer:** bagging、pasting 和随机森林的每个预测器都是独立工作的，可以通过并行实现加速训练。对于 boosting 来说，每个预测器都是基于其前序的结果，因此训练必须是有序的。对于 stacking 来说，某个指定层的预测器之间彼此独立，因而可以在多台服务器上并行训练，但是，某一层的预测器只能在其前一层的预测器全部训练完成之后，才能开始训练。

4. 包外评估的好处是什么？

bagging 集成会有约 37% 的训练样本没有参与到训练过程中，可以利用这部分样本作为验证集，检验模型的泛化能力。模型在包外评估上的性能与在测试集上的性能接近。

> **Answer:** 包外评估可以对 bagging 集成中的每个预测器使用其未经训练的实例进行评估。不需要额外的验证集，就可以对集成实施相当公正的评估。训练使用的实例越多，集成的性能可以略有提升。

5. 是什么让极端随机树比一般随机森林更加随机？这部分增加的随机性有什么用？极端随机树比一般随机森林快还是慢？

- 极端随机树划分子节点的阈值是随机确定的，进一步引入了随机性
- 这部分增加的随机性能够降低模型的方差，提升模型的泛化能力。
- 通常来说，极端随机树比一般随机森林要快。

> **Answer:** 常规决策树会搜索出特征的最佳阈值，而极限随机树直接对每个特征使用随机阈值。这种极限随机性就像是一种正则化的形式：如果随机森林对训练数据出现过度拟合，那么极限随机树可能执行效果更好。极限随机树的训练比随机森林快得多。

6. 如果你的AdaBoost集成对训练数据拟合不足，你应该调整哪些超参数？怎么调整？

可以降低学习率 learning_rate 或增加基础分类器的数量 n_estimators。

> **Answer:** 可以尝试提升估算器的数量或降低基础估算器的正则化参数，也可以尝试略微提升学习率。

7. 如果你的梯度提升集成对训练集过度拟合，你是应该提升还是降低学习率？

应该降低学习率 learning_rate，以降低正则化的程度。

> **Answer:** 应该尝试降低学习率，或者通过早期停止法来寻找合适的预测器数量。
