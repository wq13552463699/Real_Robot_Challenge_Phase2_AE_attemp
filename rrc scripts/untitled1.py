#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 09:55:13 2021

@author: qiang
"""

a = [1,2,3,4,4,4,4,5,6]

for i in range(len(a)-1,-1,-1):
    if a[i] == 4:
        a.pop(i)
        
print(a)


#%%
def func(list):
    for i in range(4,-1,-1):
        list[i+1]  = list[i]
        
    for i in range(len(list)):
        print(list[i],end='')
        
list = ['A','B','C','D','E','F']
print(func(list))

#%%

a = [1,5,7,8,2,3,3,5]
b = sorted(a)

for i in len(a):
    if a[i] != b[i]:
        
        
def check(a,b,i):
    if a[]
    
