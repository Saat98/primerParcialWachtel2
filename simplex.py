# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:09:29 2018

@author: SaulAlvarez
"""

import numpy as np

def one_simplex_step_blend(A,b,c,B,N):
    c_B=np.array([c[i] for i in B])
    c_N=np.array([c[i] for i in N])
    A_N=np.transpose(np.array([A[:,i] for i in N]))
    A_B=np.transpose(np.array([A[:,i] for i in B]))
    A_B_inv=np.linalg.inv(A_B)
    lambda_costos=np.transpose(np.dot(np.transpose(c_B),A_B_inv))
    
    r_N=np.transpose(np.dot(np.transpose(lambda_costos),A_N)-np.transpose(c_N))
    
    if(max(r_N)<=0):
        return ("Solución",np.dot(A_B_inv,b),np.dot(np.transpose(lambda_costos),b),B,N)
    
    entrada=-1
    for i in range(len(r_N)):
        if r_N[i]>0:
            entrada=N[i]
            break
        
    h=np.dot(A_B_inv,b)
    H_e=np.dot(A_B_inv,A[:,entrada])
    
    if(max(H_e)<=0):
        return ("No acotado")
    
    salida=-1
    for i in range(len(h)):
        if H_e[i]>0:
            salida=B[i]
            break
    
    for i in range(len(B)):
        if B[i]==salida:
            B[i]=entrada
    
    for i in range(len(N)):
        if N[i]==entrada:
            N[i]=salida
         
    return ("Continúa",B,N)

def mSimplexFaseII(A,b,c):
    basica=len(b)
    A=np.hstack((A,np.eye(basica)))
    c=np.hstack((c,np.zeros(basica)))
    
    N=np.arange(0,A.shape[1]-basica)
    B=np.arange(A.shape[1]-basica,A.shape[1])
    
    conteo=0
    while True:
        conteo=conteo+1
        receiver=one_simplex_step_blend(A,b,c,B,N)
        
        if receiver[0]=="Solución":
            x=np.zeros(A.shape[1])
            for i in range(len(B)):
                x[B[i]]=receiver[1][i]
                
            if min(x)<0 or max(abs(np.dot(A,np.transpose(x))-b))>1e-12:
                return(None,None,-1,conteo)
            
            return(x[0:basica],receiver[2],0,conteo)
        elif receiver[0]=="No acotado":
            return (None,None,1,conteo)
 
        B=receiver[1]
        N=receiver[2]
