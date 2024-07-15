#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[ ]:


'''
Find gradient of a [x], [y] data paired 2 lists
'''
import random
def find_gradient(x, y):
    if (len(x) != len(y)):
        assert("Dimensions does not match")
    lst_size = len(x)
    gradients = {}
    for i in range(1, lst_size-1):
        m1 = 0
        m2 = 0
        rnd_int = i
        x1 = x[rnd_int]
        x2 = x[rnd_int + 1]
        x3 = x[rnd_int - 1]
        m1 = (y[rnd_int + 1] - y[rnd_int])     / (x2 - x1)
        m2 = (y[rnd_int    ] - y[rnd_int - 1]) / (x1 - x3)
        if (round(m1,2) == round(m2,2)):
            if (m1 > 0):
                gradients[round(m1,2)] = i
    gradients = ({k: v for k, v in sorted(gradients.items(), key=lambda item: item[1])})
    return list(gradients.keys())



# In[ ]:


'''Function to convert polar Z to rect Z
'''
def pol2complex(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return complex(x, y)

'''Function to convert polar [Z] to rect [Z]'''
def phase2complex_array(arr):
    new_arr = [pol2complex(1,x) for x in arr]
    return new_arr

'''Function to convert rect [Z] to polar [Z]
   and clean the angle'''
import cmath
def complexarr2phase(arr):
    phase_arr = [cmath.phase(x) for x in arr]
    for i in range(len(phase_arr)):
        while(phase_arr[i] < 0):
            phase_arr[i] += 2 * cmath.pi
    return phase_arr

'''Function to clean polar [Z] angle'''
def cleanphase(arr):
    for i in range(len(arr)):
        while arr[i] < 0:
            arr[i] += 2 * cmath.pi
        
        while arr[i] > 2*cmath.pi:
            arr[i] -= 2 * cmath.pi
    
    return arr

'''Function to convert Rad [Z] to Degree [Z]'''
def convert_rad2deg(arr, round_val = False):
    if round_val == False:
        return [180 * x / cmath.pi  for x in arr]
    else:
        return [round(180 * x / cmath.pi, 1)  for x in arr]

'''Function to convert Degree [Z] to Rad [Z]'''
def convert_deg2rad(arr, round_val = False):
    if round_val == False:
        return [cmath.pi * x / 180 for x in arr]
    else:
        return [round(cmath.pi * x / 180, 1) for x in arr]

'''Function to place all elements of [vec] on unit circle'''
def normalize_cmplx(vec):
    rslt = [ z/abs(z) for z in vec]
    return rslt


# In[4]:


def show_voltages(title_lst, vec, figsize):
    if len(title_lst) != len(vec):
        assert("Dimensions does not match")
    
    dim = len(vec)
    figure, axis = plt.subplots(dim, figsize=figsize)
    
    
    for i in range(dim):
        title = title_lst[i]
        x = [x for x in range(0, len(vec[i]))]
        axis[i].plot(x, vec[i])
        axis[i].title.set_text(title)
        
    plt.show()                       
    
# def show_gradients(title_lst, vec):
#     vec = np.array(vec)
#     for i in range(len(vec)):
#         grad = find_gradient( [x for x in range(len(vec[i]))], vec[i])
#         print(f'{title[i]} : {grad}')
        


# In[ ]:


'''
Cosine similarity of [v2] relative to [v1]
'''
def similarity(v1, v2):
    if (len(v1) != len(v2)):
        assert("Dimensions do not match!")
    v1_phase = complexarr2phase(v1)
    v2_phase = complexarr2phase(v2)
    error_v = np.array([ math.cos(a-b) 
                        for a, b in zip(v1_phase, v2_phase) ])
    return np.sum(error_v) / len(v1)

