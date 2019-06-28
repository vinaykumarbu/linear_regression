# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 20:25:24 2019

@author: Akshay
"""

x=[1,2,3,4,5]
y=[3,4,2,4,5]
#print(x,y)

###Manual Liner Regression#####
import numpy as np

xmean=np.mean(x)
ymean=np.mean(y)

#print("Xmean:", xmean, "Ymean:", ymean)

xmin= x-xmean
ymin= y-ymean
xymul= xmin*ymin
xysum=np.sum(xymul)

xsquare=xmin**2

xsq_sum=np.sum(xsquare)

m=xysum/xsq_sum
#print(m)
c=ymean-m*xmean
#print(c)

######Squared error####
ypred=[]

for i in x:
    yp=m*i+c
    ypred.append(yp)

#print(ypred)
r_numer=(ypred-ymean)**2
sum_r_num=np.sum(r_numer)
#print(sum_r_num)

r_denom=(y-ymean)**2
sum_r_denom=np.sum(r_denom)
#print(sum_r_denom)

r2=sum_r_num/sum_r_denom
#print(r2)


######Sklearn Liner regression######
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error as mse
xx=[[1],[2],[3],[4],[5]]

#l=len(xx)
#X=xx.reshape((l,1))
model=LinearRegression()
model.fit(xx,y)
ypred=model.predict(xx)

mserr=mse(y,ypred)
rmserr=np.sqrt(mserr)
r2score=model.score(xx,y)

print("Sklearn: ",r2score, "Manual:" ,r2,) 