# **DeepLearning with Pytorch** 

## 01_Linear Model
1.过拟合、泛化

2.linear model : 
$$\hat{y}=x*w$$

3.误差评估：

针对单个sample：
$$loss=(\hat{y}-y)^2$$

针对整个training set——Mean Square Error(**MSE**):
$$1/N*\sum_{n=1}^{N}(\hat{y_n}-y_n)$$
main target:寻找最小的w使得mean loss最小

## 02_Gradient Descent
1.Optimization Problem: $w^*=arg min cost(w)$
局部最优（非突函数），引出梯度下降算法
Gradient:
$$\frac{\partial cost}{\partial w}$$
Update:
$$w=w-a\frac{\partial cost}{\partial w}$$
(a:学习率)

带入线性模型：

$$w=w-a\frac{1}{N}\sum_{n=1}^{N}2*x_n*(x_n*w-y_n)$$


鞍点问题：陷入鞍点问题无法继续迭代（鞍点：沿某些方向函数值上升，某些方向函数值下降）

解决办法：随机梯度下降（*SGD*,
 Stochastic Gradient Descent）
$$w=w-a\frac{\partial loss}{\partial w}$$

引入随机噪声则有可能跨越鞍点前往最优值

优缺点比较：
1.梯度下降：并行运算，效率更高

2.随机梯度下降：迭代，效率低，但性能更好

一种折中的算法：
Mini-Batch：批量的随机梯度下降，每次使用一组样本去求相应的梯度

## 03_Back Propagation
a two layer neural network:
 $$\hat{y}=W_2(W_1*X+b_1)+b2$$
 (weight:$W_1$,Bias:$b_1$)

 First Layer:$W_1*X+b_1$ 

 Second Layer:$W_2(...)+b_2$

 激活函数$\sigma$的引入：上文提出的两层神经网络是线性的，最终都可以化为一层，相当于只做了一层变换，所以需要引入激活函数（Nonlinear Function）变成非线性。

 forward:算每个变量的预测值，保存局部梯度

 backward:从output往input推，计算梯度（对于文中的linear model来说，是拿到$\frac{\partial loss}{\partial w}$，从而更新weight）

 in pytorch，tensor contains data and grad

 ## 04 Linear Regression with PyTorch

### PyTorch Use Guide
1.Prepare dataset 

2.Design model using Class(for $\hat{y}$ )

3.Construct loss and optimizer(using PyTorch API)

4.Training Cycle(forward(for loss),backward(for gradient),update(for new weight))

#### the first part-----prepare dataset

using mini-batch fashion

#### the second part-----Design model

construct calculate graph---Linear Unit
利用input:x和output：$\hat{y}$确定w和b的维度

## 05_Logistic Regression
分类问题思路：输出预测的并非等于某个值，而是属于某个分类的概率

*预测输出的具体值到预测概率的转变*

Logistic Function: $$ \sigma(x)=\frac{1}{1+e^{-x}}$$
