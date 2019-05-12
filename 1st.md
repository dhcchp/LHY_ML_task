1. 了解什么是Machine learning

类似于泛函，通过输入数据找到一个合适的函数。



2. 学习中心极限定理，学习正态分布，学习最大似然估计

2.1 推导回归Loss function

E=sum((y-yhat).^2). yhat=WX+b. Find the suitable X to make E smallest.


2.2 学习损失函数与凸函数之间的关系

凸函数的切线方向是损失函数的迭代方向。（？）

2.3 了解全局最优和局部最优

函数有多个凹谷时，普通迭代可能会陷入局部最优，一般和初始值的选取有关。

3. 学习导数，泰勒展开

3.1 推导梯度下降公式

<img align="center" src="figs/GradientDescent.png" width="450" alt="sota">

3.2 写出梯度下降的代码


4. 学习L2-Norm，L1-Norm，L0-Norm

4.1 推导正则化公式


4.2 说明为什么用L1-Norm代替L0-Norm


4.3 学习为什么只对w/Θ做限制，不对b做限制
