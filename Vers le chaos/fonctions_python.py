
# -*-coding:Latin-1 -*

import numpy as np
import random
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

color = [ 'royalblue', 'crimson', 'green', 'y' ]

def time_series(t, x):

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 6))

    #--> Plot the first coordinate.
    ax[0].plot(t, x[:, 0], color='royalblue')
    ax[0].plot(t[-3000:], x[-3000:, 0], color='crimson')
    ax[0].set_ylabel('$x(t)$')
    ax[0].locator_params(axis='y', nbins=5)

    #--> Plot the second coordinate.
    ax[1].plot(t, x[:, 1], color='royalblue')
    ax[1].plot(t[-3000:], x[-3000:, 1], color='crimson')
    ax[1].set_ylabel('$y(t)$')
    ax[1].locator_params(axis='y', nbins=5)

    #--> Plot the third coordinate.
    ax[2].plot(t, x[:, 2], color='royalblue')
    ax[2].plot(t[-3000:], x[-3000:, 2], color='crimson')
    ax[2].set_ylabel('$z(t)$', labelpad=18)
    ax[2].set_xlabel('t')
    ax[2].locator_params(axis='y', nbins=5)

    return

def phase_space(x):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')

    ax.plot(x[:, 0], x[:, 1], x[:, 2], color='royalblue', alpha=0.75)
    ax.plot(x[-3000:, 0], x[-3000:, 1], x[-3000:, 2], color='crimson')
    ax.set_xlabel('$x(t)$', labelpad=18)
    ax.set_ylabel('$y(t)$', labelpad=18)
    ax.set_zlabel('$z(t)$', labelpad=9)
    ax.locator_params(axis='both', nbins=5)

    return

def compare_time_series(t, x0, x1):

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 6))

    #--> Plot the first coordinate.
    ax[0].plot(t, x0[:, 0], color='royalblue')
    ax[0].plot(t, x1[:, 0], color='crimson')
    ax[0].set_ylabel('$x(t)$')
    ax[0].locator_params(axis='y', nbins=5)

    #--> Plot the second coordinate.
    ax[1].plot(t, x0[:, 1], color='royalblue')
    ax[1].plot(t, x1[:, 1], color='crimson')
    ax[1].set_ylabel('$y(t)$')
    ax[1].locator_params(axis='y', nbins=5)

    #--> Plot the third coordinate.
    ax[2].plot(t, x0[:, 2], color='royalblue')
    ax[2].plot(t, x1[:, 2], color='crimson')
    ax[2].set_ylabel('$z(t)$', labelpad=18)
    ax[2].set_xlabel('t')
    ax[2].locator_params(axis='y', nbins=5)

    return

def compare_time_series_bis(t, x0, x1):

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(12, 6))

    #--> Plot the first coordinate.
    ax[0].semilogy(t, abs(x0[:, 0]-x1[:, 0]), color='royalblue')
    ax[0].set_ylabel(r'$\| x_0(t) - x_1(t) \|$')

    #--> Plot the second coordinate.
    ax[1].semilogy(t, abs(x0[:, 1]-x1[:, 1]), color='royalblue')
    ax[0].set_ylabel(r'$\| y_0(t) - y_1(t) \|$')
    ax[1].set_xlim(0, 250)

    return

def suite_logistique(r, niter=100):

    x_val = np.zeros((niter+1,))
    x = random.uniform(0, 1)
    x_val[0] = x
    i = 0
    while i < niter and x < 1:
        x = r * x * (1 - x)
        i += 1
        x_val[i] = x

    if x < 1:
        return x, x_val
    else:
        return -1

def bifurcation_diagram(r, ntrials=50):
    """
    Cette fonction retourne (jusqu'à) *ntrials* valeurs d'équilibre
    pour les *r* d'entrée.  Elle renvoit un tuple:

    + le premier élément est la liste des valeurs prises par le paramètre *r*
    + le second est la liste des points d'équilibre correspondants
    """

    r_v = []
    x_v = []
    for rr in r:
        j = 0
        while j < ntrials:
            xx, _ = suite_logistique(rr,niter=250)
            if xx > 0:  # A convergé: il s'agit d'une valeur d'équilibre
                r_v.append(rr)
                x_v.append(xx)
            j += 1                      # Nouvel essai

    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    ax.plot(r_v, x_v, ',')
    ax.set_xlabel('$r$')
    ax.set_ylabel('$x$')

    return

def plot_suite(x):

    fig = plt.figure(figsize=(12, 2))
    ax = fig.gca()

    #--> Plot the first coordinate.
    ax.plot(x, color='royalblue')
    ax.plot(x, color='crimson', marker='.', linestyle='')
    ax.set_ylabel('$x_n$')
    ax.set_xlabel('$n$')
    ax.locator_params(axis='y', nbins=5)
