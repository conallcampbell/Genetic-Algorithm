#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:02:35 2023

@author: conallcampbell
"""

import numpy
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

#kronecker product code for more than 2 inputs
def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result

# Building the adjacency matrix for a 6-qubit star network
# Each on-site energy is the same, therefore w_ii = 0 for all diagonal elements
A = np.array([
    [0 , 1 , 1 , 1 , 1 , 1],
    [1 , 0 , 0 , 0 , 0 , 0],
    [1 , 0 , 0 , 0 , 0 , 0],
    [1 , 0 , 0 , 0 , 0 , 0],
    [1 , 0 , 0 , 0 , 0 , 0],
    [1 , 0 , 0 , 0 , 0 , 0]
    ])

# We choose an initial state for the system
rho0 = np.array([[1] , [0] , [0] , [0] , [0] , [0]])
state = kron(rho0,np.conjugate(rho0.T))

# We observe how this changes as a function of time
# thus, we choose a time-evolution operator U = exp(-iHt) for H = A
def U(t):
    return expm(complex(0 , 1) * A * t)

# Defining the evolution of the state
def rho1(t):
    return U(t) @ state @ np.conjugate(U(t).T)

# Now we make a plot of the evolution of the diagonal elements of rho1

ts = numpy.linspace(0, 10, 100)
diagonal_elements_00 = [rho1(t)[0,0] for t in ts]
diagonal_elements_11 = [rho1(t)[1,1] for t in ts]
diagonal_elements_22 = [rho1(t)[2,2] for t in ts]
diagonal_elements_33 = [rho1(t)[3,3] for t in ts]
diagonal_elements_44 = [rho1(t)[4,4] for t in ts]
diagonal_elements_55 = [rho1(t)[5,5] for t in ts]




##########################################
## Plotting
# ps = numpy.linspace(0.0, 1.0, 1000)

# figure, axis = plt.subplots(1, 1)
figure, ax1 = plt.subplots(1,1)
ax1.set_xlabel("t")
ax1.set_ylabel("Evolution of probability")
ax1.plot(ts , diagonal_elements_00)
figure.savefig('evolution_00_star.png',dpi=300)

# figure, axis = plt.subplots(1, 1)
figure, ax2 = plt.subplots(1,1)
ax2.set_xlabel("t")
ax2.set_ylabel("Evolution of probability")
ax2.plot(ts , diagonal_elements_11)
figure.savefig('evolution_11_star.png',dpi=300)

# figure, axis = plt.subplots(1, 1)
figure, ax3 = plt.subplots(1,1)
ax3.set_xlabel("t")
ax3.set_ylabel("Evolution of probability")
ax3.plot(ts , diagonal_elements_22)
figure.savefig('evolution_22_star.png',dpi=300)

# figure, axis = plt.subplots(1, 1)
figure, ax4 = plt.subplots(1,1)
ax4.set_xlabel("t")
ax4.set_ylabel("Evolution of probability")
ax4.plot(ts , diagonal_elements_33)
figure.savefig('evolution_33_star.png',dpi=300)

# figure, axis = plt.subplots(1, 1)
figure, ax4 = plt.subplots(1,1)
ax4.set_xlabel("t")
ax4.set_ylabel("Evolution of probability")
ax4.plot(ts , diagonal_elements_33)
figure.savefig('evolution_44_star.png',dpi=300)

# figure, axis = plt.subplots(1, 1)
figure, ax4 = plt.subplots(1,1)
ax4.set_xlabel("t")
ax4.set_ylabel("Evolution of probability")
ax4.plot(ts , diagonal_elements_33)
figure.savefig('evolution_55_star.png',dpi=300)