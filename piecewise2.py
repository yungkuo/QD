# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:10:49 2013

@author: KyoungWon
"""

def piecewise2(indx,value):
    from pylab import *
    result=np.zeros(indx[-1]+1)
    for i in range(indx.size-1):
        if type(value[i])==str:
            value[i]=0
            for j in range(int(indx[i]), int(indx[i+1])):
                result[j]=value[i-1]+(value[i+1]-value[i-1])/(indx[i+1]-indx[i])*(j-indx[i])
                #result[indx[i]:indx[i+1]]=value[i-1]
        else:
            result[indx[i]:indx[i+1]]=value[i]
    result[indx[i+1]]=value[i]
    result=np.array(result)
    return result

