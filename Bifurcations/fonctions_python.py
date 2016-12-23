
# -*-coding:Latin-1 -*

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec

color = [ 'royalblue', 'crimson', 'green', 'y' ]

def phase_line_plot(x, f):

    """

    Trace l'espace des phases pour un systeme dynamique du premier ordre.

    INPUTS:
    ------

    x : Discretisation de l'espace des phases 1D.
        one-dimensional numpy array.

    f : Fonction f(x) caracterisant le systeme dynamique.

    """

    #---> Creation de la figure.
    fig, axs = plt.subplots(1, 3, sharex=True)
    fig.set_size_inches(12, 4)

    c = [-0.5, 0, 0.5]

    for ax, c in zip(axs, c):

    #--> Trace la courbe dx/dt = f(x).
        ax.plot(x, f(x, c))
        if (c < 0):
            ax.set_title(r'$\mu < 0$', y=-.25)
        elif (c == 0):
            ax.set_title(r'$\mu = 0$', y=-.25)
        elif (c > 0):
            ax.set_title(r'$\mu > 0$', y=-.25)

    #-------------------------------------#
    #----- Mise en page de la figure -----#
    #-------------------------------------#

    #--> Bornes inferieure et superieure pour l'axe des x.
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 1)
    #--> Label de l'axe des x.
        ax.set_xlabel(r'$n$')
        ax.set_xticks([-1, 0, 1])
        for label in ax.get_xticklabels():
            label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))

    #--> Positionnement de ce label sur la figure.
        ax.xaxis.set_label_coords(1.05, 0.5)

    #--> Label de l'axe des y.
        ax.set_ylabel(r'$\dot{n}$', rotation=0)
    #--> Positionnement de ce label.
        ax.yaxis.set_label_coords(0.5, 1.05)
    #--> Pas de ticks le long de l'axe y.
        ax.set_yticks([], [])

    #--> Remplace le cadre habituel de la figure par un systeme d'axes
    #    centre en (0, 0).
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data',0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data',0))

    #--> Ajout des fleches au bout des axes.
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

    # get width and height of axes object to compute
    # matching arrowhead length and width
        dps = fig.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height

    # manual arrowhead width and length
        hw = 1./50.*(ymax-ymin)
        hl = 1./50.*(xmax-xmin)
        lw = 1. # axis line width
        ohg = 0.25 # arrow overhang

    # compute matching arrowhead length and width
        yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
        yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
        ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
                head_width=hw, head_length=hl, overhang = ohg,
                length_includes_head= True, clip_on = False)

        ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
                head_width=yhw, head_length=yhl, overhang = ohg,
                length_includes_head= True, clip_on = False)

    return

def transcritical_bifurcation():

    x_sub = np.linspace(-1, 0, 100)
    x_sup = np.linspace(0, 1, 100)

    fig = plt.figure()
    fig.set_size_inches(12, 4)
    ax = fig.gca()

    ax.plot(x_sub, 0*x_sub, color=color[0])
    ax.plot(x_sup, 0*x_sup, '--', color=color[0])

    ax.plot(x_sub, x_sub, '--', color=color[0])
    ax.plot(x_sup, x_sup, color=color[0])

    ax.set_xlabel(r'$\mu$')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    ax.set_ylabel(r'$x^*$')
    ax.set_title(r'$\dot{x} = \mu x - x^2$')

    return

def saddle_node_bifurcation():

    mu = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True)
    fig.set_size_inches(16, 4)

    #--> dx/dt = mu - x**2
    ax[0].plot(mu, np.sqrt(mu), color=color[0])
    ax[0].plot(mu, -np.sqrt(mu), '--', color=color[0])
    ax[0].set_title(r'$\dot{x} = \mu - x^2$')
    ax[0].set_xlabel(r'$\mu$')
    ax[0].set_ylabel(r'$x^*$', rotation=0)
    ax[0].set_xticks([-1, 0, 1])
    ax[0].set_yticks([-1, 0, 1])

    #--> dx/dt = -mu + x**2
    ax[1].plot(mu, -np.sqrt(mu), color=color[0])
    ax[1].plot(mu, np.sqrt(mu), '--', color=color[0])
    ax[1].set_title(r'$\dot{x} = -\mu + x^2$')
    ax[1].set_xlabel(r'$\mu$')

    #--> dx/dt = -mu - x**2
    ax[2].plot(-mu, np.sqrt(mu), color=color[0])
    ax[2].plot(-mu, -np.sqrt(mu), '--', color=color[0])
    ax[2].set_title(r'$\dot{x} = -\mu - x^2$')
    ax[2].set_xlabel(r'$\mu$')

    #--> dx/dt = mu + x**2
    ax[3].plot(-mu, -np.sqrt(mu), color=color[0])
    ax[3].plot(-mu, np.sqrt(mu), '--', color=color[0])
    ax[3].set_title(r'$\dot{x} = \mu + x^2$')
    ax[3].set_xlabel(r'$\mu$')


    return

def pitchfork_bifurcation():

    mu = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_size_inches(16, 4)

    #--> dx/dt = mu x - x**3
    ax[0].plot(mu, np.sqrt(mu), color=color[0])
    ax[0].plot(mu, -np.sqrt(mu), color=color[0])
    ax[0].plot(-mu, 0*mu, color=color[0])
    ax[0].plot(mu, 0*mu, '--', color=color[0])
    ax[0].set_title(r'$\dot{x} = \mu x - x^3$ (supercritique)')
    ax[0].set_xlabel(r'$\mu$')
    ax[0].set_ylabel(r'$x^*$', rotation=0)
    ax[0].set_xticks([-1, 0, 1])
    ax[0].set_yticks([-1, 0, 1])

    #--> dx/dt = mu x + x**3
    ax[1].plot(-mu, np.sqrt(mu), '--', color=color[0])
    ax[1].plot(-mu, -np.sqrt(mu), '--', color=color[0])
    ax[1].plot(-mu, 0*mu, color=color[0])
    ax[1].plot(mu, 0*mu, '--', color=color[0])
    ax[1].set_title(r'$\dot{x} = \mu x + x^3$ (sous-critique)')
    ax[1].set_xlabel(r'$\mu$')

    return

def pitchfork_bifurcation_bis():


    alpha = 0.25
    mu = np.linspace(0, 2, 200)
    mu_lower = np.linspace(-1/(4*alpha), 0, 200)
    mu_upper = np.linspace(-1/(4*alpha), 2, 200)

    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = fig.gca()

    #--> dx/dt = mu x + x**3
    mu_1 = 0*mu
    mu_2 = np.sqrt((1+np.sqrt(1+4*alpha*mu_upper))/(2*alpha))
    mu_3 = np.sqrt((1-np.sqrt(1+4*alpha*mu_lower))/(2*alpha))

    ax.plot(-mu, 0*mu, color=color[0])
    ax.plot(mu, 0*mu, '--', color=color[0])
    ax.plot(mu_upper, mu_2, color=color[0])
    ax.plot(mu_upper, -mu_2, color=color[0])
    ax.plot(mu_lower, mu_3, '--', color=color[0])
    ax.plot(mu_lower, -mu_3, '--', color=color[0])
    ax.set_title(r'$\dot{x} = \mu x + x^3 - 0.25 x^5$')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$x^*$', rotation=0, labelpad=10)

    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([], [])

    return

def pitchfork_bifurcation_impropre():

    mu = np.linspace(-2, 2, 100)
    x = np.linspace(-2, 2, 100)
    mu, x = np.meshgrid(mu, x)

    f = (1./27) + mu*x - x**3

    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = fig.gca()

    ax.contour(mu, x, f, levels=[0.], colors=color[0])
    ax.set_title(r'$\dot{x} = \alpha + \mu x - x^3$')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$x^*$', rotation=0, labelpad=10)

    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([], [])

    return


def streamlines(ax, x, y, u, v, density=1):

    #-->
    magnitude = np.sqrt(u**2 + v**2)

    ax.streamplot(x, y, u, v, color=magnitude, cmap=plt.cm.inferno, density=density)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return


def phase_plot_2d(x, y, f):

    #---> Creation de la figure.
    fig, axs = plt.subplots(1, 3, sharex=True)
    fig.set_size_inches(12, 4)

    mu = [-0.5, 0, 0.5]

    for ax, r in zip(axs, mu):

        xdot = np.zeros_like(x)
        ydot = np.zeros_like(y)
        xdot[:], ydot[:] = f(x[:], y[:], r)
        streamlines(ax, x, y, xdot, ydot, density=1.)

        if (r < 0):
            ax.set_title(r'$\mu < 0$', y=-.25)
        elif (r == 0):
            ax.set_title(r'$\mu = 0$', y=-.25)
        elif (r > 0):
            ax.set_title(r'$\mu > 0$', y=-.25)


        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.set_xlabel(r'$x$')

    axs[0].set_ylabel(r'$y$', rotation=0, labelpad=12)
    return

def phase_plot_homoclinic(f):

    fig = plt.figure()
    fig.set_size_inches(12, 8)
    ax = fig.gca()

    #---> Points fixes.
    x_0 = np.array([0., 0.])
    x_1 = np.array([1., 0.])

    t = np.linspace(0, 20, 1000)

    x0 = x_0.copy()
    x0[0] += 0.001
    sol = odeint(f, x0, t)
    ax.plot(sol[:, 0], sol[:, 1], color=color[1], lw=2)

    x0 = x_0.copy()
    x0[0] -= 0.001
    sol = odeint(f, x0, t)
    ax.plot(sol[:, 0], sol[:, 1], color=color[1], lw=2)

    x0 = x_0.copy()
    x0[0] -= 0.001
    x0[1] += 0.001
    sol = odeint(f, x0, -t[:250])
    ax.plot(sol[:, 0], sol[:, 1], color=color[1], lw=2)

    x0 = x_0.copy()
    x0[0] += 0.001
    x0[1] -= 0.001
    sol = odeint(f, x0, -t[:550])
    ax.plot(sol[:, 0], sol[:, 1], color=color[1], lw=2)

    x0 = x_1.copy()
    x0[0] += 0.25
    sol = odeint(f, x0, t)
    ax.plot(sol[:, 0], sol[:, 1], '--', color=color[1], lw=1)

    #-->
    x = np.linspace(-2, 2)
    x, y = np.meshgrid(x, x)
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f([x[:], y[:]], 0)
    streamlines(ax, x, y, xdot, ydot, density=0.75)


    ax.plot(x_0[0], x_0[1], 'o', ms=10, color=color[0])
    ax.plot(x_1[0], x_1[1], 'o', ms=10, color=color[0])


    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0, labelpad=12)

    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])

    return

def saddle_node_of_periodic_orbits(mu=0.1):

    f = lambda r, mu: mu*r + r**3 - r**5
    r = np.linspace(0, 2)

    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = fig.gca()

    ax.plot(r, f(r, mu))
    ax.plot(r, 0*r, 'k--')

    ax.set_ylim(-.5, .5)
    ax.set_yticks([-.5, 0, .5])
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\dot{r}$', rotation=0, labelpad=12)

    return
