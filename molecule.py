# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 16:59:12 2018
@author: Roland Scheidel
"""
import numpy as np
import unit_conv
from scipy.interpolate import interp1d
import helper_functions
import os

class Molecule:
    '''
     generic class for a molecule, can be extended by a class for a specific molecule
    '''
    def __init__(self, name, electronic_state, rot_state, m1,m2, Re, ome, Ed = 0, min_electronic_energy =0, full_name=''):
        '''
        :param name: formula name of the molecule
        :param electronic_state: short Name of the electronic state for referencing,
         needs to match file name for the RKR Data
        :param rot_state: rotational quantum number J
        :param m1: Mass of the first atom
        :param m2: Mass of second atom
        :param Re: equilibrium distance
        :param ome: vibrational constant
        :param Ed: Dissociation energy
        :param min_electronic_energy: relative to the minimum of the electronic ground state
        :param full_name: full name of the electronic_state,to be used for plots etc
        '''
        self.name = name
        self.full_name = full_name
        self.electronic_state = electronic_state
        self.rot_state = rot_state
        self.m1 = m1
        self.m2 = m2
        self.M = m1*m1/(m1+m2)
        self.Re = Re
        self.ome = ome
        self.load_RKR_potential(electronic_state)
        self.Ed = Ed
        self.min_electronic_energy = min_electronic_energy

    def load_RKR_potential(self, electronic_state):
        RKR_data = np.genfromtxt(os.path.join('Data',self.name,'States',electronic_state+'_RKR.txt'))
        self.RKR_xi = unit_conv.angst_to_a0(np.append(RKR_data[::-1, 1], RKR_data[1:, 2]))
        self.RKR_Vi = unit_conv.cm_to_hatree2(np.append(RKR_data[::-1, 0], RKR_data[1:, 0]))
        self.RKR_Vi = self.RKR_Vi - unit_conv.cm_to_hatree2(RKR_data[0,0])
        self.RKR_Vi += self.rot_state * (self.rot_state + 1) / (2 * self.M * self.RKR_xi**2)

    def RKR_linspace(self, n=512):
        '''
        crates a linspace over the full space where data for the state is available
        :param n:
        :return:
        '''
        xi = np.linspace(min(self.RKR_xi), max(self.RKR_xi), n)
        return xi

    def RKR_potential(self, xi, interpolation="quadratic"):
        RKR_xi = self.RKR_xi
        RKR_Vi = self.RKR_Vi

        if max(xi) > max(self.RKR_xi):
            raise ValueError('Maximum x value outside of range of numerical potential')

        if min(xi) < min(self.RKR_xi):
            raise ValueError('Minimum x outside of range of numerical potential')

        fint = interp1d(RKR_xi, RKR_Vi, interpolation)
        return fint(xi)
    def RKR_raw(self):
        return self.RKR_xi, self.RKR_Vi

    def harmonic(self, x):
        harmonic = 0.5 * self.M * (self.ome ** 2) * (x - self.Re) ** 2
        # V=0.5*self.M*(x-self.Re)**2
        return harmonic


    def morse(self, xi):
            a = self.ome / np.sqrt(2 * self.Ed / self.M)
            exponent = -a * (xi - self.Re)
            V = self.Ed * (1 - np.exp(exponent)) ** 2
            return V