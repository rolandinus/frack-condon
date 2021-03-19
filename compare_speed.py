import numpy as np
from Na2 import Na2
import helper_functions as helpers
from time import perf_counter
state = Na2("X", 0)
max_grid = 15
times_full_matrix = []
times_tridiagonal = []
for counter in range(7, max_grid):
    N = 2 ** counter
    print('test mit '+ str(N) + ' Gridpunkten:')
    xi =  state.RKR_linspace(N)
    Vi = state.RKR_potential(xi)

    t1 = 0
    t2 = 0
    repetitions = 3 #since there are runtime fluctuations depending on other system tasks, take the average
    for repeat in range(1,repetitions):
        t1_start = perf_counter()
        helpers.solve_schroedinger_full_matrix(xi, Vi, Na2.M)
        t1_stop = perf_counter()
        t1 = t1 + t1_stop - t1_start

        t2_start = perf_counter()
        helpers.solve_schroedinger_tridiagonal(xi, Vi, Na2.M)
        t2_stop = perf_counter()
        t2 = t2 + t2_stop - t2_start
    t1 = t1 / repetitions
    times_full_matrix.append(t1)
    t2 = t2 / repetitions
    times_tridiagonal.append(t2)
    print("   benötigte Zeit mit kompletter Matrix: " + str(t1))
    print("   benötigte Zeit mit nur diagonal Elementen: " + str(t2))







