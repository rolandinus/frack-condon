# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 18:44:46 2018

@author: Roland Scheidel
"""
import numpy as np
from Na2 import Na2
import helper_functions as helpers
import seaborn
import matplotlib.pyplot as plt
#import sys

# number of grid points
import os

print(os.environ['PATH'])

state_1 = Na2("X", 0)
state_2 = Na2("B", 0)

plt.figure()
helpers.compare_state_potentials([state_1, state_2, Na2("B", 0)])
plt.show()
N = 2 ** 12
print('Verwende '+ str(N) + 'Gridpunkte')
#sys.exit()
# maximum vibrational level to be considered:
max_v = 10

# construct a spacial grid xi which range includes both Potentials, returning also the limits for which the potential is avalaible
xi, limits_1, limits_2 = helpers.double_linspace(state_1.RKR_xi, state_2.RKR_xi, N)
print('Das entspricht einer räumlichen Auflsösung von a0', xi[1]-xi[0])
xi_1 = xi[limits_1[0]:limits_1[1]]  # spacial grid vector for the range of available data for the state_1 potential
xi_2 = xi[limits_2[0]:limits_2[1]]  # same for the state_2 potential

# get the interpolated potentials
Vi_1 = state_1.RKR_potential(xi_1)
# Vi_1 = state_1.morse(xi_1)
Vi_1_harmonic = state_1.harmonic(xi_1)
Vi_2 = state_2.RKR_potential(xi_2)
# Vi_2 = state_2.morse(xi_2)
Vi_2_harmonic = state_2.harmonic(xi_2)

# solve the stationary SEQs
ev_1_rkr, U_ = helpers.solve_schroedinger_tridiagonal(xi_1, Vi_1, Na2.M)
ev_1_morse, U_1_morse = helpers.solve_schroedinger_tridiagonal(xi, state_1.morse(xi), Na2.M)
ev_2_rkr, U2_ = helpers.solve_schroedinger_tridiagonal(xi_2, Vi_2, Na2.M)
ev_2_morse, U_2_morse = helpers.solve_schroedinger_tridiagonal(xi, state_2.morse(xi), Na2.M)

# paste in the solutions for both states in a vector that spans over the range of both potentials
U = np.zeros([N, max_v])
U[limits_1[0]:limits_1[1], 0:max_v] = U_[:, 0:max_v]

U2 = np.zeros([N, N])
U2[limits_2[0]:limits_2[1], 0:max_v] = U2_[:, 0:max_v]

P = U ** 2
P2 = U2 ** 2

FC_numeric = np.zeros([max_v, max_v])
FC_morse = np.zeros([max_v, max_v])
FC_harmonic = np.zeros([max_v, max_v])


# calculate the franck condon factors numerical and analyticaly harmonic potential

for j in range(0, max_v):
    for i in range(0, max_v):
        s_rkr = helpers.simpson(xi, U[:, i] * U2[:, j]) ** 2
        s_morse = helpers.simpson(xi, U_1_morse[:, i] * U_2_morse[:, j]) ** 2
        FC_numeric[i, j] = s_rkr
        FC_morse[i, j] = s_morse
        FC_harmonic[i, j] = helpers.franck_condon_analytic(state_1, i, state_2, j)


plt.figure(1)

helpers.plot_trans(xi, xi_1, Vi_1, xi_2, Vi_2, U[:, 0], U2, ev_1_rkr, ev_2_rkr, FC_numeric[:, 0], state_1, state_2)
plt.draw()
plt.show()

helpers.compare_model_functions(xi_1, state_1)
helpers.compare_model_functions(xi_2, state_2)

# Show FC factors
plt.figure()
ax1 = seaborn.heatmap(FC_numeric, linewidth=0.02, annot=True, fmt=".2f", vmin=0, vmax=FC_numeric.max(), cbar=False)
ax1.xaxis.set_ticks_position('top')
ax1.xaxis.set_label_text("V'' (" + state_1.full_name + ")")
ax1.xaxis.set_label_position('top')
ax1.yaxis.set_label_text("V' (" + state_2.full_name + ")")
ax1.yaxis.set_label_position('left')
ax1.set_title('')
plt.savefig(os.path.join('Output', 'FC_numeric_' + state_1.electronic_state + '_' + state_2.electronic_state + '_.svg'),
            bbox_inches='tight', format='svg')

# Compare Solutions with model functiosn


'''
Compare errors in the FC Factors of the model functions (harmonic potential, morse potential) with the spectroscopic RKR potential
'''
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(16, 8))

err_morse = abs(FC_numeric - FC_morse) * 100
err_harmonic = abs(FC_numeric - FC_harmonic) * 100

max_err = (max(err_morse.max(), err_harmonic.max()))
seaborn.set(font_scale=1.25)
seaborn.heatmap(err_morse, linewidth=0.02, annot=True, fmt=".2f", vmin=0, vmax=max_err, cbar=False, ax=ax1, )
ax1.xaxis.set_ticks_position('top')
ax1.xaxis.set_label_text("V''")
ax1.xaxis.set_label_position('top')
ax1.yaxis.set_label_text("V'")
ax1.yaxis.set_label_position('left')
ax1.set_title('Morse Potential')

seaborn.heatmap(err_harmonic, linewidth=0.02, annot=True, fmt=".2f", vmin=0, vmax=max_err, cbar=False, ax=ax2)
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_text("V''")
ax2.xaxis.set_label_position('top')
ax2.yaxis.set_label_text("V'")
ax2.yaxis.set_label_position('left')
ax2.set_title('Harmonisches Potential')

# seaborn.heatmap(abs(FC3[0:10,0:10]), linewidth=0.05, annot=True,fmt=".2f")
plt.savefig(os.path.join('Output',
                         'error_morse_vs_harmonic_' + state_1.electronic_state + '_' + state_2.electronic_state + '_.svg'),
            bbox_inches='tight', format='svg')
filename = os.path.join('Output',
                        'error_morse_vs_harmonic.svg' + '_' + state_1.electronic_state + '_' + state_2.electronic_state + '_transition.svg')
plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, format='svg')

plt.show()
