# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 13:13:53 2026

@author: PC

"""
def fibonacci(n):
    a = 0
    b = 1
    fib = []
    while a < n:
        fib.append(a)
        a, b = b, a+b
     
    return fib


liste = fibonacci(50)


def classer(classeur, nombre):
    
    if nombre >= 0 :
         classeur['positif'].append(nombre)
    else: 
         classeur['negatif'].append(nombre)
    return classeur

