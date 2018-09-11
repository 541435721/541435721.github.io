---
layout: post
title: 'Learning Deep Features for Discriminative Localization'
subtitle: '利用类别标签让网络具备目标定位能力'
date: 2018-09-11
categories: paper
# cover: 'http://pics.qiangbenyk.cn/2018_09_11_09_41_50_212-Q.png'
tags: Attention object_location
---

# Learning Deep Features for Discriminative Localization

## 动机

卷积网络在计算机视觉任务上效果显著。但是大多数任务往往解决图像中有什么，无法直接获取图像中目标的位置信息或者目标之间的关系。本文提出的方法，可以用图像的类别标签获得目标在图像中的位置信息。该技术可以用到精细分类，定位，和知识发现等任务中。

## 方法

- global average pool

  对整张图像做平均池化

  ![](http://pics.qiangbenyk.cn/2018_09_11_10_20_29_122-b.png)

- class activation map

  特定类别的激活映射可以反映用于该类别判定的特定区域。

  分类网络会对图像特征进行加权求和，得到最后结果。同样，使用这些加权的权重值，对global average pool的输入进行加权，可以得到特定类别的类激活映射图。

  详细方法：

  给定一张图像，$f_k(x,y)$表示最后一层卷积层的$k$个通道特征在空间坐标$(x,y)$处的值。对这$k$个通道做global average pool，$F^k$表示$\sum_{x,y}{f_k(x,y)}$，对于给定的类别c，图像的softmax输入为$S_c$为$\sum_{k}{w_k^cF_k}$，其中$w_k^c$是对应类别$c$和特征通道$k$的权重。本质上来说，$w_k^c$表示第$k$个通道特征对$c$类别判定的重要性。$c$类的softmax输出为$P_c$为$\frac{exp(S_c)}{\sum_{c}{exp(S_c)}}$，这里不使用偏置量。

  对于特定类别$c$的类激活映射可以表示为$M_c(x,y)=\sum_{k}{w_k^cf_k(x,y)}$ ，表示每个空间网格$(x,y)$对于类别$c$判断的重要程度。

  ![](http://pics.qiangbenyk.cn/2018_09_11_10_54_24_771-j.png)

- 将网络中的全连接替换为GAP后跟一个全连接

## 实验

![](http://pics.qiangbenyk.cn/2018_09_11_11_17_09_612-6.png)

![](http://pics.qiangbenyk.cn/2018_09_11_11_17_34_094-8.png)



