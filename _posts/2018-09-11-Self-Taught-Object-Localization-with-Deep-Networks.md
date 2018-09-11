---
layout: post
title: 'Self-Taught Object Localization with Deep Networks'
subtitle: '利用卷积网络让网络具备目标定位能力'
date: 2018-09-11
categories: paper
# cover: 'http://pics.qiangbenyk.cn/2018_09_11_09_41_50_212-Q.png'
tags: Attention object_location deeplearning
---

# Self-Taught Object Localization with Deep Networks

## 动机

目标识别可以细分为两个子任务，(1)整张图片的分类，(2)检测。目标检测可以为分类任务提供很多好处，包括图像中的目标定位，排除干扰信息提升鲁棒性。但是检测任务需要大量的边框标签信息，这类标注获取难度较大。

## 方法

- 原理：对原始输入图像增加掩膜，如果掩膜覆盖到影响类别判断的区域，则类别判断的得分也会随之变化。最终，在原图上会获得一组对类别得分变化较大的子窗口。

  ![](http://pics.qiangbenyk.cn/2018_09_11_13_09_37_890-c.png)

  第一行为原始图像，判断为金鱼的概率为0.999；第二行中对金鱼部分增加掩膜，得出来图像为金鱼的概率为0.002，得分变化较大，可以估计该区域为金鱼所在区域。

- 输入掩膜

  假定有一个网络模型$f:R^N \rightarrow R^C$，表示将有$N$个像素的图像$x \in R^N$映射到一个$C$类的置信向量$y \in R^C$上。置信向量$y$表示为$y=[y_1,y_2,…y_C]^T$，$y_i$表示对应的第$i$类的分类得分。这里对图像给定矩形区域$b=[b_x,b_y,w,h]\in N^4$中像素值用一个三维向量$g$做替换，$g$表示图像的三个通道。$b_x,b_y$分别表示矩形框起始点坐标，$w,h$表示矩形框的宽度和高度。$g$从训练数据中学习，并被设置为图像三通道的均值。对于给定图像$x$和掩膜矩阵$b$以及替换值$g$，掩膜可以函数表示为$h_g:R^N\times N^4\rightarrow R^N$。输出图像和输入图像大小相等。将掩膜前后的得分变化定义为$\delta_f(x,b)=max(f(x),f(h_g(x,b)),0)$。两个差值越大，说明该区域对于该类的判别性越强。

  如果已知某个图像$x$以及类别标签$c$，可以定义退化函数$d_{CL}:R^N\times N^4 \rightarrow R$，$d_{CL}(x,b)=\delta_f(x,b)^T\mathbb{I}_c$，其中$\mathbb{I}_c\in N^C$是一个指示向量，在第$c$个分量上为1，其余为0。使用退化函数可以计算指定类别的目标区域。

  如果图像的类别信息未知，则使用预测模型对整张图片的$C_I$个类别判断定义$d_{WL}(x,b)=\delta_f(x,b)^T\mathbb{I}_{top-C_I}$，其中$\mathbb{I}_{top-C_I}$也是一个类别指示向量，排名前$C_I$位置的分量为1，其余为0。本文使用`top-5`做计算。

- 聚类

  本文使用一种迭代方法，贪心比较可选区域，每一次迭代合并两个可以最大化相似函数的区域，直到只有一个区域的时候停止迭代。

  两个区域相似性度量：

  > 如果两个区域满足一下几个情况，则可以认为包含了相同的目标：
  >
  > 1. 在分类得分上造成近似的得分退化
  >
  >    $s_{drop}=(x,b_i,b_j)=1-abs(d_m(x,b_i)-d_m(x,b_j))=max(1-d_m(x,b_i),1-d_m(x,b_j))$,其中$m \in {CL,WL}$
  >
  > 2. 外部表现相似
  >
  >    $s_{app}(x,b_i,b_j)=z(\phi(x,b_i),\phi(x,b_j))$，其中$z(·,·)$表示灰度直方图交叉相似度，$\phi(·,·)$表示特征描述子，最后一层全连接层的输出，或者softmax的输入。
  >
  > 3. 覆盖区域都很大
  >
  >    $s_{size}(x,b_i,b_j)=1-\frac{size(b_i)+size(b_j)}{size(x)}​$
  >
  > 4. 在空间上距离较近
  >
  >    $s_{fill}(x,b_i,b_j)=1-\frac{size(b_i\cup b_j)-size(b_i)-size(b_j)}{size(x)}$,其中$b_i \cup b_j$表示包含$b_i$和$b_j$的边框。
  >
  > 整体的相似度得分为$s(b_i,b_j,x)=\sum_{l\in L}\alpha_ls_l(b_i,b_j,x)$，其中$L=[drop,app,size,fill]$，试验中设置四个项的权值相等，



