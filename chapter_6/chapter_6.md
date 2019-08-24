# Chapter 6: Decision Tree

This chapter briefly introduces the basic principles of Decision Tree and how to use it in sklearn. This markdown file contains my own solutions to exercise of the book, from Ex.1 to Ex.6. For codes, refer to the jupyter notebook under the same directory,

> For further details on Decision Tree, including branch clipping, regularization, etc, please refer to other machine learninf tutorials. This book may only contains how to use Decision Tree with sklearn simply.

****

1. 如果训练集有 100 万个实例，训练决策树（无约束）大致的深度是多少？

决策树的深度取决于特征数量。

> **Answr：** 一个包含 m 个叶节点的均衡二叉树（每一层节点数翻倍）的深度等于 $\log_2(m)$ 的四舍五入。通常来说，二元决策树训练到最后大体都是平衡的，**如果不加以限制，最后平均每个叶节点一个实例**。因此，如果训练集包含一百万个实例，那么决策树深度约等于 $\log_2(10^6) \approx 20$ 层。

2. 通常来说，子节点的基尼不纯度是高于还是低于其父节点？是通常更高/更低，还是永远更高/更低？

决策树的生长是根据节点的基尼不纯度进行的，每一次决策后，子节点的基尼不纯度永远低于其父节点。

> **Answer：** CART训练算法分裂每个节点的方法，就是使其子节点的基尼不纯度的加权和最小，因此通常来说子节点的基尼不纯度低于其父节点。（但也有例外情况，详见书后答案。）

3. 如果决策树过度拟合训练集，减少 max_depth 是否为一个好主意？

可以通过减小 max_depth 对决策树进行正则化。

> **Answer：** 降低 max_depth 会限制模型，使其正则化。

4. 如果决策树对训练集拟合不足，尝试缩放输入特征是否为一个好主意？

决策树不需要对输入特征进行缩放，因此缩放输入特征无法对模型作出影响。

> **Answer：** 决策树的优点之一就是它们不关心训练数据是否缩放或者集中，因此缩放输入特征没有任何作用。

5. 如果在包含 100 万个实例的训练集上训练决策树需要一个小时，那么在包含 1000 万个实例的训练集上训练决策树，大概需要多长时间？

决策树的计算复杂度为$O(\log_2(m))$，在包含 100 万个实例的训练集上训练需要一个小时，那么在包含 1000 万个实例的训练集上训练需要的时间（单位：min）为
$$ 60 \times \frac{\log_2(10^7)}{\log_2(10^6)} \approx 60 $$
这说明决策树的计算效率是非常高的

> **Answer：** 决策树的训练复杂度为
> $$O(n \times m \log(m))$$
> 所以。如果将训练集大小乘以 10，训练时间将乘以 
> $$K = \frac{n \times 10m \times \log(10m)}{n \times m \log(m)} = 10 \times \frac{\log(10m)}{\log(m)}$$
> 如果 $m=10^6$，那么 $K \approx 11.7$，所以训练 1000 万个实例大约需要 11.7 个小时。

6. 如果训练集包含 100,000 个实例，设置 presort=True 可以加快训练吗？

presort 对训练数据进行预处理，在训练集较小的情况下可以加快训练，但是对于包含 100,000 个实例的训练集来说反而可能降低训练速度。

> **Answer：** 只有当数据集小雨数千个实例时，预处理训练集才可以加速训练，对于大型数据集，设置 presort=True 会显著减慢训练
