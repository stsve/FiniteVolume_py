#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:13:52 2020

@author: stefanesved
"""

import matplotlib.pyplot as plt
import numpy as np

#--------------------------------------------------------------------#
#                    Graphiques Spalart-Allmaras                     #
#--------------------------------------------------------------------#

alpha = np.array([0,3,6,9,12,15])

C_dsa = np.array([0.022321,0.0271002,0.03439,0.044749,0.061643,0.096947])
C_lsa = np.array([1.62326,2.00164,2.3365,2.59913,2.76596,2.704848])

C_dsst = np.array([0.02169,0.026558,0.03416,0.045535,0.066702,0.153072])
C_lsst = np.array([1.60998,1.98746,2.32067,2.578415,2.72227,2.52077])

alpha_exp = np.array([-2.19e-3,2.99,5.96,8.96,12,15])
C_lexp = np.array([1.66,2.04,2.4,2.76,3.06,2.02])

cdexp = np.array([2.07e-2,2.95e-2,4.15e-2])
clexp = np.array([2.41,2.87,3.14])

fig1, ax1 = plt.subplots()
ax1.plot(C_dsa,C_lsa, marker='s', markerfacecolor='black', markersize=4, color='black', linewidth=1.5,label='Spalart-Allmaras')
ax1.plot(C_dsst,C_lsst, marker='D', markerfacecolor='dimgray', markersize=4, color='dimgray', linewidth=1.5,linestyle='--',label='$k-\omega$ SST')
ax1.plot(cdexp,clexp, marker='^', markerfacecolor='blue', markersize=4, color='blue', linewidth=1.5,linestyle='-.',label='Expérimental')
ax1.axvline(C_dsa[-2],linestyle=':',color='cyan')
ax1.axvline(C_dsst[-2],linestyle=':',color='magenta')
plt.text(0.035, 1.8, r'$C_D = 0.0616$', {'color': 'c', 'fontsize': 10})
plt.text(0.068, 2, r'$C_D = 0.0667$', {'color': 'm', 'fontsize': 10})
ax1.set_title('$C_L$ - $C_D$')
ax1.set_xlabel('$C_D$')
ax1.set_ylabel('$C_L$')
plt.legend()
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(alpha,C_lsa, marker='s', markerfacecolor='black', markersize=4, color='black', linewidth=1.5,label='Spalart-Allmaras')
ax2.plot(alpha,C_lsst, marker='D', markerfacecolor='dimgray', markersize=4, color='dimgray', linewidth=1.5,linestyle='--',label='$k-\omega$ SST')
ax2.plot(alpha_exp,C_lexp, marker='^', markerfacecolor='blue', markersize=4, color='blue', linewidth=1.5,linestyle='-.',label='Expérimental')
ax2.axvline(alpha[-2],linestyle=':',color='dimgray')
plt.text(12.2, 1.85, r'$\alpha = 12^{\circ}$', {'color': 'gray', 'fontsize': 11})
ax2.set_title(r'$C_L$ - $\alpha$')
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel('$C_L$')
plt.legend()
plt.show()

