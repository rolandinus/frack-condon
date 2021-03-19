# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:42:56 2018

@author: Roland Scheidel
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os


def franck_condon_analytic(mol_1, v1, mol_2, v2):
    fc = 0
    d = mol_2.Re - mol_1.Re
    a1 = mol_1.ome * mol_1.M
    a2 = mol_2.ome * mol_2.M
    A = 2 * np.sqrt(a1 * a2) / (a1 + a2)
    S = a1 * a2 * d ** 2 / (a1 + a2)
    b1 = a2 * np.sqrt(a1) * d / (a1 + a2)
    b2 = -a1 * np.sqrt(a2) * d / (a1 + a2)

    for i in range(0, v1 + 1):
        for j in range(0, v2 + 1):
            if 0 == (i + j) % 2:
                # print("i,j",i,j)
                K = (i + j) // 2
                I = scipy.special.factorial2(2 * K - 1, True) / ((a1 + a2) ** K)
                B = scipy.special.binom(v1, i) * scipy.special.binom(v2, j)
                H1 = scipy.special.eval_hermite(v1 - i, b1)
                H2 = scipy.special.eval_hermite(v2 - j, b2)

                C = ((2 * np.sqrt(a1)) ** i) * ((2 * np.sqrt(a2)) ** j)
                fc = fc + B * H1 * H2 * C * I

    norm = A * np.exp(-S) / (2 ** (v1 + v2) * scipy.special.factorial(v1, True) * scipy.special.factorial(v2, True))

    fc = fc ** 2 * norm

    return fc


def simpson(xi, fi):
    '''
    xi: spacial grid
    fi: function values for on
    '''
    dx = xi[1] - xi[0]
    N = fi.size  # number of Points
    boundaries = fi[0] + fi[-1]
    sum_odd = np.sum(fi[1:N - 2:2], 0)
    sum_even = np.sum(fi[2:N - 2:2], 0)
    integral = dx * (1.0 / 3.0) * (boundaries + 2 * sum_even + 4 * sum_odd)
    return integral


def solve_schroedinger_full_matrix(xi, Vi, M=1, normalize=True):
    '''
    :param xi: spatial grid vector
    :param Vi: potential grid vector
    :param M: reduced_mass
    '''
    dx = xi[1] - xi[0]
    N = xi.size

    diagonal = Vi + 1.0 / (M * dx * dx)
    off_diag = np.ones(N - 1) * (-1.0 / (2 * M * dx * dx))
    H = np.diag(diagonal) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    # build the tridiagonal matrix as a sum of three matrices
    ev, U = scipy.linalg.eigh(H)

    if normalize:
        C = (1.0 / simpson(xi, U ** 2)) ** 0.5
        U = U * C
    return ev, U


def solve_schroedinger_tridiagonal(xi, Vi, M=1, normalize=True):
    '''
    :param xi: spatial grid vector
    :param Vi: potential grid vector
    :param M: reduced_mass
    :param normalize: normalize the eigenvectors?
    '''
    dx = xi[1] - xi[0]
    N = xi.size

    diagonal = Vi + 1.0 / (M * dx * dx)
    off_diag = np.ones(N - 1) * (-1.0 / (2 * M * dx * dx))

    ev, U = scipy.linalg.eigh_tridiagonal(diagonal, off_diag)

    if normalize:
        C = (1.0 / simpson(xi, U ** 2)) ** 0.5  # calculate the normalization constant for all eigenvactors
        U = U * C

    return ev, U


def sdg_nonequidistant(xi, Vi, M=1):
    N = xi.size
    H = np.zeros([N, N])

    dx1 = xi[1] - xi[0]
    dx2 = dx1
    # xi[2]-xi[1]
    p = dx2 / dx1
    q = dx1 * dx2 * (1 + p)
    B = 2 * (1 + p) / (2 * M * q) + Vi[0]
    C = -1 / (2 * M * q)
    H[0, 0] = B
    H[0, 1] = C
    for i in range(1, N - 1):
        dx1 = xi[i] - xi[i - 1]
        dx2 = xi[i + 1] - xi[i]
        p = dx2 / dx1
        q = dx1 * dx2 * (1 + p)
        A = -2 * p / (2 * M * q)
        B = 2 * (1 + p) / (2 * M * q) + Vi[i]
        C = -2 / (2 * M * q)

        H[i, i - 1] = A
        H[i, i] = B
        H[i, i + 1] = C

    dx1 = xi[N - 1] - xi[N - 2]
    dx2 = dx1
    # xi[2]-xi[1]
    p = dx2 / dx1
    q = dx1 * dx2 * (1 + p)
    A = -2 * p / (2 * M * q)
    B = 2 * (1 + p) / (2 * M * q) + Vi[N - 1]
    H[N - 1, N - 1] = B
    H[N - 1, N - 2] = A

    ev, U = np.linalg.eigh(H)
    return ev, U


def find_limit(Vi, maxval):
    # helper function to find the boundaries of a gridvector around the minimum
    # so that all Values are within a certain range
    start = np.argmin(Vi)
    i = start
    while (maxval - Vi[i]) / maxval > 0.02 and Vi[i] < maxval and i < Vi.size - 1 and i > 0:
        i = i + 1

    j = start
    while (maxval - Vi[j]) / maxval > 0.02 and Vi[i] < maxval and i < Vi.size and i > 0:
        j = j - 1
    return j, i


def plot_trans(xi, xi_1, Vi_1, xi_2, Vi_2, psi1, psi2, ev1, ev2, FC, state_1, state_2):
    '''
    :param xi:
    :param xi_1:
    :param Vi_1:
    :param xi_2:
    :param Vi_2:
    :param psi1:
    :param psi2:
    :param ev1:
    :param ev2:
    :param FC:
    :return:
    '''
    minV1 = min(Vi_1)
    minV1_index = np.where(Vi_1 == minV1)
    n = FC.size

    max_relevant_value = min(ev1[n] * 1.75, Vi_1.max() * 0.9)
    s1, e1 = find_limit(Vi_1, max_relevant_value)  # only draw relevant parts of the potentials
    max_relevant_value = min(ev2[n] * 1.75, Vi_2.max() * 0.95)
    s2, e2 = find_limit(Vi_2, max_relevant_value)

    s = np.searchsorted(xi, xi_1[s1])
    e = np.searchsorted(xi, xi_2[e2], 'right')

    scale = 0.4 * (ev1[1] - ev1[0]) / max(psi1)
    ax1 = plt.subplot(212)

    # plot first state
    plt.plot(xi_1[s1:e1], Vi_1[s1:e1], 'k-')
    plt.fill(xi[s:e], ev1[0] + scale * psi1[s:e], c='r', alpha=1.0)
    plt.axvline(x=xi[minV1_index])
    plt.xlabel('Kernabstand ($a_0$)')
    plt.ylabel("Energy ($E_h$)", horizontalalignment='left')
    ax1.annotate(state_1.full_name, (xi_1[e1], 0.9 * Vi_1[e1]))

    # plot second state
    ax2 = plt.subplot(211, sharex=ax1)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax2.annotate(state_2.full_name, (xi_2[e2] - 0.15, state_2.min_electronic_energy + 0.84 * Vi_2[e2]))

    Vi_2 = Vi_2 + state_2.min_electronic_energy
    ev2 = ev2 + state_2.min_electronic_energy
    plt.plot(xi_2[s2:e2], Vi_2[s2:e2], 'k-')

    plt.subplots_adjust(hspace=0.01)
    plt.axvline(x=xi[minV1_index])
    for i in range(0, n):
        alpha = 0.1 + 0.9 * FC[i] / max(FC)
        print('alpha', alpha)
        plt.fill(xi[s:e], ev2[i] + scale * psi2[s:e, i], c='r', alpha=min(0.1 + 0.9 * FC[i] / max(FC), 1))
        print(FC[i])
    filename = os.path.join('Output',
                            'wavefunctions_' + state_1.name + '_' + state_1.electronic_state + '_' + state_2.electronic_state + '.svg')
    plt.savefig(filename, bbox_inches='tight', format='svg')
    return


def compare_state_potentials(state_list):
    for state in state_list:
        xi = state.RKR_linspace()
        Vi = state.RKR_potential(xi) + state.min_electronic_energy
        plt.plot(xi, Vi)
        raw_xi, raw_vi = state.RKR_raw()
        plt.plot(raw_xi, raw_vi + state.min_electronic_energy, 'k.', markersize=2.5)
        plt.annotate(state.full_name, [xi[-1] - 0.5, Vi[-1] - 0.0075])
        plt.axhline(y=state.Ed + state.min_electronic_energy, c='black', linestyle=':', linewidth='1')

    plt.xlabel('Kernabstand ($a_0$)')
    plt.ylabel("Energy ($E_h$)", horizontalalignment='left')
    filename = os.path.join('Output', 'compare_potentials_of_states.svg')
    plt.savefig(filename, bbox_inches='tight', format='svg')
    return


def compare_model_functions(xi, molecule):
    ax = plt.figure()
    plt.plot(xi, molecule.harmonic(xi), 'r-')
    plt.plot(xi, molecule.morse(xi), 'b-')
    plt.plot(xi, molecule.RKR_potential(xi), 'g-')
    ax.legend(['Harmonisch', 'Morse', 'RKR'], loc='upper left', bbox_to_anchor=(0.125, 0.87))
    plt.xlabel('Kernabstand ($a_0$)')
    plt.ylabel("Energy ($E_h$)", horizontalalignment='left')
    filename = os.path.join('Output', 'compare_model_functions_' + molecule.electronic_state + '_state.svg')
    plt.savefig(filename, format='svg')
    return


def double_linspace(range_a, range_b, n):
    '''
    helper function to create a linpspace for two overlapping intervals
    :param range_a: min and max of the first grid vector
    :param range_b: min and max of the second grid vector
    :param n: total number of grid points
    :return: grid vector and indizes for range_a, range_b,
    '''
    x_min = min(range_a[0], range_b[0])
    x_max = max(range_a[-1], range_b[-1])
    xi = np.linspace(x_min, x_max, n)
    a_min = np.searchsorted(xi, range_a[0], side='right')
    b_min = np.searchsorted(xi, range_b[0], side='right')

    a_max = np.searchsorted(xi, range_a[-1])
    b_max = np.searchsorted(xi, range_b[-1])
    a_indices = [a_min, a_max]
    b_indices = [b_min, b_max]

    return xi, a_indices, b_indices
