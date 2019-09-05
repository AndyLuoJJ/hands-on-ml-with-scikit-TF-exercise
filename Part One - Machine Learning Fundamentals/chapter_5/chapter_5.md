# Chapter 5 Support Vector Machine

This chapter introduces the basics of SVM(Support Vector Machine). Further detail of it can be found in other books. This book only focuses on how to use SVM with scikit-learn and how to roughly improve its performance. 

> This file is my solution to the exercises. Coding exercises can be found in the jupyter notebook rather than this markdown file.

****

1. 支持向量机的基本思想是什么？

支持向量机通过支持向量确定决策面，使支持向量的间隔尽可能大。
                              
> **Answer:** 支持向量机的基本思想是拟合类别之间可能的、最宽的“街道”。换言之，它的目的是**使决策边界之间的间隔最大化**，从而分隔出两个类别的训练实例。SVM执行软间隔分类时，实际上是在完美分类和拟合最宽街道之间进行妥协，即*允许少数实例落在街道上*。还有一个关键点是在训练非线性数据时，记得使用核函数（为模型引入非线性）。

2. 什么是支持向量？

距离决策面最近的实例称为支持向量，它们决定了决策面。

> **Answer:** 支持向量机训练完成后，位于“街道”上的实例称为支持向量，这也包括处于边界上的实例。**决策边界完全由支持向量决定**。非支持向量的实例完全没有任何影响，只要它们在街道之外，就不会对决策边界产生任何影响。

3. 使用SVM时，对输入值进行缩放为什么重要？

如果不进行缩放，在某个特征上的SVM的斜率可能趋近于零，对分类性能造成影响。

> **Answer:** 如果训练集不经过缩放，SVM将趋于忽略值较小的特征。

4. SVM分类器在对实例进行分类时，会输出信心分数吗？概率呢？

SVM在进行分类时不会输出信心分数和概率，在决策面确定以后，SVM计算实例对于决策面的符号，根据符号直接分类。

> **Answer:** SVM能够输出测试实例与决策边界之间的距离，可以将其用作信心分数，但是这个分数不能直接转化成类别概率的估算。在创建SVM时，可以在scikit-learn中设置```probability=True```，在训练完成后，算法将使用逻辑回归对SVM分数进行校准，从而得到概率值。

5. 如果训练集有上千万个实例和几百个特征，你应该使用SVM原始问题还是对偶问题来训练模型？

实例和特征数量较多的情况下，应该使用对偶问题，

> **Answer:** 这个问题仅适用于线性SVM，因为**核SVM只能使用对偶问题**。对于SVM来说，**原始形式的计算复杂度与训练实例的数量成正比，而其对偶形式的计算复杂度与某个介于$m^2$和$m^3$之间的数量成正比**。因此在训练实例数量非常大的时候，需要使用原始问题，因为对偶问题会非常慢。

6. 假设你用RBF核训练了一个SVM分类器，看起来似乎对训练集拟合不足，你应该提升还是降低gamma？C呢？

应该提升gamma值。C也一样。C的值控制松弛系数对模型的影响程度，C值越大，软间隔越大，允许犯错的实例数量越多，能够起到减轻过拟合的作用。

> **Answer:** RBF核支持向量机对训练集拟合不足，可能是由于过度正则化导致的，应该提升gamma或C（或同时提升二者）来降低正则化。
