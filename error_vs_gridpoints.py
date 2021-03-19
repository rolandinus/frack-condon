"""
Created on Fri Aug 10 18:44:46 2018
@author: Roland Scheidel
"""
from Na2 import Na2
import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

import os
a = 1
b = 13

X = Na2("X", 0)
B = Na2("B", 0)
maxV = 2
average_absolute_error = []
h = []
maxSteps = 14

# calculate FC factors analytically
FC_analytic = np.zeros([maxV, maxV])
for i in range(0, maxV):
    for j in range(0, maxV):
        FC_analytic[i, j] = hf.franck_condon_analytic(X, i, B, j)

# compare with numerical calculations for varying number of grid points
for counter in range(4, maxSteps):
    print(counter)
    N = 2 ** counter
    xi = np.linspace(a, b, N)
    h.append(N)

    Vi_X = X.harmonic(xi)
    Vi_B = B.harmonic(xi)

    ev, U = hf.solve_schroedinger_tridiagonal(xi, Vi_X, Na2.M)
    ev2, U2 = hf.solve_schroedinger_tridiagonal(xi, Vi_B, Na2.M)

    P = U ** 2
    P2 = U2 ** 2

    FC_numeric = np.zeros([maxV, maxV])
    t1_start = perf_counter()
    for i in range(0, maxV):
        for j in range(0, maxV):
            s = hf.simpson(xi, U[:, i] * U2[:, j]) ** 2
            FC_numeric[i, j] = s
            FC_analytic[i, j] = hf.franck_condon_analytic(X, i, B, j)

    t1_stop = perf_counter()

    absolute_error = abs(FC_analytic - FC_numeric) 
    print('err')
    print(absolute_error)
    average_absolute_error.append(np.average(absolute_error))
plt.figure(2)
plt.plot(h, average_absolute_error, '-+')
print('max err',h, average_absolute_error)

plt.yscale('log', basey=10)
plt.xscale('log', basex=2)
plt.xlabel('Anzahl an Gridpunkten')
plt.ylabel('durchschnittlicher absoluter Fehler')

filename = os.path.join('Output', 'gridpoints_vs_error_'+str(maxSteps)+'.svg')
plt.savefig(filename, bbox_inches='tight', format='svg')


plt.show()


