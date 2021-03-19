# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 11:07:32 2017

@author: Roland
"""


def m2nat(m):
    return m / 1.97327e-7


def wavenumber_to_frequency(cm):
    return 137.036 / (cm_to_a0(1 / cm))


def angst2nat(a):
    return a / 1.97327e3


def angst_to_a0(a):
    return a * 1.8897259886


def cm_to_a0(cm):
    return cm * 188972613.392125


def au2nat(au):
    return au * 5.29177 * (10 ** (-11)) / (1.97327 * 10 ** (-7))


def au2m(au):
    return au * 5.29177e-11


def au2a(au):
    return au * 0.529177


def kg2nat(kg):
    return kg / 1.78266E-36


def ev2j(ev):
    return ev * 1.60218E-19


def j2ev(j):
    return j / 1.60218E-19


def cm2j(cm):
    return 1.98645e-23 * cm


def cm2ev(cm):
    return cm * 1.98645e-4 / 1.60218


def ev2hatree(ev):
    return 0.0367493 * ev


def cm_to_hatree2(cm):
    return ev2hatree(cm2ev(cm))


def cm_to_hatree(cm):
    return cm * 4.55633e-6


def ev2cm(ev):
    return ev * 1.60218 / 1.98645e-4


def s2nat(s):
    return s / 6.58212e-16


def molecular_weight_to_me(g_per_mol):
    return g_per_mol / 6.02214e23 / 9.1093897e-28


def g_to_me(g):
    return g / 9.1093897e-28
