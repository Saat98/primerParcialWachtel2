# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:19:03 2018

@author: SaulAlvarez
"""

import time
import simplex
import numpy as np

def generaKleeMinty(m):
    c=-np.ones(m)
    A=np.zeros((m,m),dtype=float)
    b=np.array([(2**(i+1))-1 for i in range(m)],dtype=float)
    
    for i in range(m):
        A[i,i]=1
        for j in range(i):
            A[i,j]=2
            
    return(A,b,c)

def SimplexKleeMinty(m):
    A,b,c=generaKleeMinty(m)
    
    inicio=time.time()
    resultado=simplex.mSimplexFaseII(A,b,c)
    final=time.time()
    
    return(resultado[1],resultado[3],final-inicio)
    
for i in range(3,11):
    print(SimplexKleeMinty(i))