#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:55:00 2018
load base_hiver
@author: jbrlod
"""
import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
#from netCDF4 import Dataset
#Name of the base to load
fname = "base_hiver_2008.pklgz"


with gzip.open(fname,"rb") as fp:
    dictio = pickle.load(fp)

print("MAZIGH")

SSTMW = dictio['SSTMW']
SSTMW = SSTMW[:, :, :92]
#image to plot
imind = 45
plt.imshow(SSTMW[imind])
plt.show()

imind = 55
plt.imshow(SSTMW[imind])
plt.show()

imind = 65
plt.imshow(SSTMW[imind])
plt.show()
