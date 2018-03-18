# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:09:29 2018

@author: SaulAlvarez
"""

import numpy as np
import scipy as sp

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
        return ("No acotado",None)
    
    salida=-1
    cociente=1000000000
    for i in range(len(h)):
        if H_e[i]>0 and h[i]/H_e[i]<cociente:
            cociente=h[i]/H_e[i]
            salida=B[i]
    
    for i in range(len(B)):
        if B[i]==salida:
            B[i]=entrada
    
    for i in range(len(N)):
        if N[i]==entrada:
            N[i]=salida
         
    return ("Continúa",B,N)

def one_simplex_step_blend_nuevo(A,b,c,B,N,L,U):
    c_B=np.array([c[i] for i in B])
    c_N=np.array([c[i] for i in N])
    A_N=np.transpose(np.array([A[:,i] for i in N]))
    A_B=np.transpose(np.array([A[:,i] for i in B]))
    
    lambda_costos=sp.linalg.solve_triangular(L,sp.linalg.solve_triangular(U,c_B,trans='T'),trans='T')
    
    r_N=np.transpose(np.dot(np.transpose(lambda_costos),A_N)-np.transpose(c_N))
    
    if(max(r_N)<=0):
        return ("Solución",np.dot(np.linalg.inv(A_B),b),np.dot(np.transpose(lambda_costos),b),B,N)
    
    entrada=-1
    for i in range(len(r_N)):
        if r_N[i]>0:
            entrada=N[i]
            break
        
    h=sp.linalg.solve_triangular(U,sp.linalg.solve_triangular(L,b))
    H_e=sp.linalg.solve_triangular(U,sp.linalg.solve_triangular(L,A[:,entrada]))
    
    if(max(H_e)<=0):
        return ("No acotado",None)
    
    salida=-1
    cociente=1000000000
    for i in range(len(h)):
        if H_e[i]>0 and h[i]/H_e[i]<cociente:
            cociente=h[i]/H_e[i]
            salida=B[i]
    
    m=len(B)
    for i in range(m):
        if B[i]==salida:
            B[i:m-1]=B[i+1:]
            U[:,i:m-1]=U[:,i+1:]
            B[-1]=entrada
            U[:,-1]=sp.linalg.solve_triangular(L,A[:,entrada])
            print('Salió la columna',i)
    
    for i in range(len(N)):
        if N[i]==entrada:
            N[i:len(N)-1]=N[i+1:]
            N[-1]=salida
    
    for i in range(m-1):
        if abs(U[i][i])<1e-12:
                M=np.eye(m)
                M[i][i+1]=1
                U=np.dot(M,U)
                M[i][i+1]=-1
                L=np.dot(L,M)
        
        M=np.eye(m)
        M[i][i]=1/U[i][i]
        U=np.dot(M,U)
        M[i][i]=1/M[i][i]
        L=np.dot(L,M)
        
        if abs(U[i+1][i])>1e-12:
            M=np.eye(m)
            M[i+1][i]=-U[i+1][i]
            U=np.dot(M,U)
            M[i+1][i]=-M[i+1][i]
            L=np.dot(L,M)
            
    print('U nueva:',U,sep='\n')
    
    return ("Continúa",B,N,L,U)

def mSimplexFaseII_nuevo(A,b,c):
    basica=len(b)
    A=np.hstack((A,np.eye(basica)))
    c=np.hstack((c,np.zeros(basica)))
    
    N=np.arange(0,A.shape[1]-basica)
    B=np.arange(A.shape[1]-basica,A.shape[1])
    
    A_B=np.transpose(np.array([A[:,i] for i in B]))
    P,L,U=sp.linalg.lu(np.transpose(A_B))
    P=np.linalg.inv(np.transpose(A_B))
    L,U=np.transpose(U),np.transpose(L)
    B_aux=np.dot(np.transpose(B),P)
    for i in range(len(B)):B[i]=int(B_aux[i])
    
    conteo=0
    while True:
        conteo=conteo+1
        print('B:',B,'N:',N,'L:',L,'U:',U,sep='\n')
        print('LU:',np.dot(L,U),sep='\n')
        print('A_B:',np.transpose(np.array([A[:,i] for i in B])),sep='\n')
        receiver=one_simplex_step_blend_nuevo(A,b,c,B,N,L,U)
        
        if receiver[0]=="Solución":
            x=np.zeros(A.shape[1])
            for i in range(len(B)):
                x[B[i]]=receiver[1][i]
            
            return(x[0:basica],receiver[2],0,conteo)
        elif receiver[0]=="No acotado":
            return (None,None,1,conteo)
 
        B=receiver[1]
        N=receiver[2]
        L=receiver[3]
        U=receiver[4]
        
        if conteo%50==0:
            A_B=np.transpose(np.array([A[:,i] for i in B]))
            P,L,U=sp.linalg.lu(np.transpose(A_B))
            P=np.inv(np.transpose(A_B))
            L,U=np.transpose(U),np.transpose(L)
            B_aux=np.dot(np.transpose(B),P)
            for i in range(len(B)):B[i]=int(B_aux[i])

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
            
            return(x[0:basica],receiver[2],0,conteo)
        elif receiver[0]=="No acotado":
            return (None,None,1,conteo)
 
        B=receiver[1]
        N=receiver[2]
