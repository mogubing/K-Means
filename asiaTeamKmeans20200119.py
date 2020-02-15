# -*- coding: utf_8
_author_ = 'huihui.gong'
_date_ = '2020/1/19'
# 步骤：
# 1、首先随机选取k个中心点
# 2、然后每个点分配到最近的类中心点（欧式距离），形成k个类，然后重新计算每个类的中心点（均值）
# 3、重复第二步，直到类不发生变化，或者你也可以设置最大迭代次数，这样即使类中心点发生变化，但是只要达到最大迭代次数就会结束。
# 此原始数据没有结果数据。
# ?如何验证正确性？
# knn与kmeans区别：
# 1、knn是分类算法，kmeans是聚类算法
# 2、两个算法是两种不同的学习方式：kmeans是非监督学习，不需要事先给出分类标签。knn是监督学习，需要我们给出训练数据的分类标识。
# 3、kmeans的k代表k个分类，knn的k代表k个最接近的邻居
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
asiaTeamData=pd.read_csv('D:\\PycharmProjects\\actualProject1116\\decisiontree200114\\datas\\asiaTeamData.csv',sep=',')
# print(asiaTeamData)
x_train=asiaTeamData[['2019年国际排名','2018世界杯','2015亚洲杯']]
print(x_train)
ss=MinMaxScaler()
x_train=ss.fit_transform(x_train)
teamclf=KMeans(n_clusters=3,init='k-means++',precompute_distances='auto')
teamclf.fit(x_train)
y_predict=teamclf.predict(x_train)
print(y_predict)
print(pd.Series(y_predict,name='echelon'))
df_x_train=pd.DataFrame(x_train)
teamResult=pd.merge(df_x_train,pd.Series(y_predict,name='聚类'),how='left',left_index=True,right_index=True)
print(teamResult)



