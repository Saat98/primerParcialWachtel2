# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:09:29 2018

@author: SaulAlvarez
"""

import numpy as np

def one_simplex_step_blend(A,A_N,A_B,A_B_inv,b,c,B,N):
    c_B=np.array([c[i] for i in B])
    c_N=np.array([c[i] for i in N])
    lambda_costos=np.transpose(np.dot(np.transpose(c_B),A_B_inv))
    
    r_N=np.transpose(np.dot(np.transpose(lambda_costos),A_N)-np.transpose(c_N))
    
    if(max(r_N)<=0):
        return ("Solución",np.dot(A_B_inv,b),np.dot(np.transpose(lambda_costos),b))
    
    entrada=-1
    for i in range(len(r_N)):
        if r_N[i]>0:
            entrada=i
            break
        
    h=np.dot(A_B_inv,b)
    H_e=np.dot(A_B_inv,A_N[:,entrada])
    
    if(max(H_e)<=0):
        return ("No acotado")
    
    salida=-1
    for i in range(len(h)):
        if H_e[i]>0:
            salida=i
            break
    
    for i in range(len(B)):
        if B[i]==salida:
            B[i]=entrada
    
    for i in range(len(N)):
        if N[i]==entrada:
            N[i]=salida
            
    A_B=A_B+np.dot(A_N[:,entrada]-A_B[:,salida],np.transpose(np.ones(A_N.shape[1])))
    A_B_inv=A_B_inv-(np.linalg.multi_dot([A_B_inv,A_N[:,entrada]-A_B[:,salida],np.transpose(np.ones(A_N.shape[1])),A_B_inv]))/(1+np.linalg.multi_dot([np.transpose(np.ones(A_N.shape[1])),A_B_inv,A_B_inv,A_N[:,entrada]-A_B[:,salida]]))
    
    return ("Continúa",A_N,A_B,A_B_inv,B,N)

def mSimplexFaseII(A,b,c):
    N=np.arange(1,A.shape[1])
    B=np.arange(A.shape[1]+1,A.shape[1]+len(b))
    A_N=np.array([A[:,i] for i in N])
    A_B=np.array([A[:,i] for i in B])
    A_B_inv=np.linalg.inv(A_B)
    
    basica=len(b)
    A=np.hstack((A,np.eye(basica)))
    c=np.hstack((c,np.zeros(basica)))
    
    conteo=0
    while True:
        conteo=conteo+1
        receiver=one_simplex_step_blend(A,A_N,A_B,A_B_inv,b,c,B,N)
        
        if receiver[0]=="Solución":
            if min(receiver[1])<0 or np.dot(A,receiver[1])!=b:
                return(None,None,-1,conteo)
            
            return(receiver[1],receiver[2],0,conteo)
        elif receiver[0]=="No acotado":
            return (None,None,1,conteo)
        
        A_N=receiver[1]
        A_B=receiver[2]
        A_B_inv=receiver[3]
        B=receiver[4]
        N=receiver[5]
