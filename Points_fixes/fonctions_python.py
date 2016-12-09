 # -*-coding:Latin-1 -*

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec

params = {'text.usetex' : True,
          'font.size' : 18,
          'font.family' : 'lmodern',
          'text.latex.unicode' : True}
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
plt.rcParams['axes.linewidth'] = 1
plt.rcParams.update(params)
color = [ 'blue', 'orange', 'green', 'y' ]

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
    fig = plt.figure()
    fig.set_size_inches(18, 6)
    ax = fig.gca(aspect=3)

    #--> Trace la courbe dx/dt = f(x).
    ax.plot(x, f)
    #--> Place les points d'equilibre.
    eq_stab = np.arange(-np.pi, 2*np.pi, 2*np.pi)
    eq_unstab = np.arange(-2*np.pi, 3*np.pi, 2*np.pi)
    ax.plot(eq_stab, 0.*eq_stab, 'o', ms=12, color=color[1])
    ax.plot(eq_unstab, 0.*eq_unstab, 's', ms=12, color=color[2])

    #-------------------------------------#
    #----- Mise en page de la figure -----#
    #-------------------------------------#

    #--> Bornes inferieure et superieure pour l'axe des x.
    ax.set_xlim(-8, 8)
    #--> Label de l'axe des x.
    ax.set_xlabel(r'$x$')
    #--> Positionnement de ce label sur la figure.
    ax.xaxis.set_label_coords(1.05, 0.5)
    #--> Choix des ticks et de leur label le long de l'axe x.
    ax.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_xticklabels([r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$'])
    #--> Petit carre semi-transparent autour des graduations.
    for label in ax.get_xticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))

    #--> Label de l'axe des y.
    ax.set_ylabel(r'$\dot{x}$', rotation=0)
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
    hw = 1./20.*(ymax-ymin)
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
             head_width=hw, head_length=0.25*hl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=0.25*yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    return










def streamlines_and_isoclines(x, y, u, v):

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_size_inches(10, 5)

    streamlines(ax[0], x, y, u, v)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$', rotation=0)

    isoclines(ax[1], x, y, u, v)

    plt.tight_layout()

    return

def streamlines(ax, x, y, u, v, density=1):

    #-->
    magnitude = np.sqrt(u**2 + v**2)

    ax.streamplot(x, y, u, v, color=magnitude, cmap=plt.cm.inferno, density=density)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    #ax.set_aspect('equal')
    ax.set_title(r'Streamlines')

    return

def isoclines(ax, x, y, u, v):

    ax.quiver(x[::4, ::4], y[::4, ::4], u[::4, ::4], v[::4, ::4], color='k')
    c1 = ax.contour(x, y, u, levels=[0.], colors=color[0], label=r'$\dot{x}=0$')
    c2 = ax.contour(x, y, v, levels=[0.], colors=color[1], label=r'$\dot({y}=0$')

    fmt = {}
    strs = [r'$\dot{x}=0$']
    for l, s in zip(c1.levels, strs):
        fmt[l] = s
    ax.clabel(c1, c1.levels, inline=True, fmt=fmt, fontsize=12)

    fmt = {}
    strs = [r'$\dot{y}=0$']
    for l, s in zip(c2.levels, strs):
        fmt[l] = s
    ax.clabel(c2, c2.levels, inline=True, fmt=fmt, fontsize=12)

    ax.set_xlabel(r'$x$')

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    ax.set_title(r'Isoclines')

    return








def saddle_node():

    #--> Maillage de l'espace des phases.
    x = np.linspace(-1, 1, 25)
    x, y = np.meshgrid(x, x)

    #--> Definition du systeme linearise
    f = lambda x, y: np.array([-x, -2*y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    #--> Plot le portrait de phase a proximite du point fixe.
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_size_inches(10, 5)

    streamlines(ax[0], x, y, xdot, ydot)
    ax[0].plot(0, 0, 'o', color=color[1], ms=8)
    ax[0].set_title('Noeud (stable)')

    #--> Definition du systeme linearise
    f = lambda x, y: np.array([-x, 2*y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[1], x, y, xdot, ydot)
    ax[1].plot(0, 0, 'o', color=color[1], ms=8)
    ax[1].set_title('Col')

    return










def foyer_et_centre():

    #--> Maillage de l'espace des phases.
    x = np.linspace(-1, 1, 25)
    x, y = np.meshgrid(x, x)

    #--> Definition du systeme linearise
    f = lambda x, y: np.array([-x + y, -x - y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    #--> Plot le portrait de phase a proximite du point fixe.
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_size_inches(10, 5)

    streamlines(ax[0], x, y, xdot, ydot)
    ax[0].plot(0, 0, 'o', color=color[1], ms=8)
    ax[0].set_title('Foyer (stable)')

    #--> Definition du systeme linearise
    f = lambda x, y: np.array([-y, x])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[1], x, y, xdot, ydot)
    ax[1].plot(0, 0, 'o', color=color[1], ms=8)
    ax[1].set_title('Centre')

    return









def noeud_impropre():

    #--> Maillage de l'espace des phases.
    x = np.linspace(-1, 1, 25)
    x, y = np.meshgrid(x, x)

    #--> Definition du systeme linearise
    f = lambda x, y: np.array([-x , -3*x - y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    #--> Plot le portrait de phase a proximite du point fixe.
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_size_inches(10, 5)
    streamlines(ax[0], x, y, xdot, ydot)
    ax[0].plot(0, 0, 'o', color=color[1], ms=8)
    ax[0].set_title('Avec terme seculaire')

    #--> Definition du systeme linearise
    f = lambda x, y: np.array([-x, -y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])
    streamlines(ax[1], x, y, xdot, ydot)
    ax[1].plot(0, 0, 'o', color=color[1], ms=8)
    ax[1].set_title('Sans terme seculaire')

    return










def upper_figure(ax):

    #--> Maillage de l'espace des phases.
    x = np.linspace(-1, 1, 10)
    x, y = np.meshgrid(x, x)

    #--> Définition du système linéarisé
    f = lambda x, y: np.array([-x + y, -x + -y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[0], x, y, xdot, ydot, density=0.5)
    ax[0].plot(0, 0, 'o', color=color[1], ms=4)
    ax[0].set_title('Foyer attractif', fontsize=14)
    ax[0].set_xticks([], [])
    ax[0].set_yticks([], [])

    #--> Définition du système linéarisé
    f = lambda x, y: np.array([y, -x])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[1], x, y, xdot, ydot, density=0.5)
    ax[1].plot(0, 0, 'o', color=color[1], ms=4)
    ax[1].set_xticks([], [])
    ax[1].set_yticks([], [])
    ax[1].set_title('Centre', fontsize=14)

    #--> Définition du système linéarisé
    f = lambda x, y: np.array([x + y, -x + y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[2], x, y, xdot, ydot, density=0.5)
    ax[2].plot(0, 0, 'o', color=color[1], ms=4)
    ax[2].set_title('Foyer repulsif', fontsize=14)
    ax[2].set_xticks([], [])
    ax[2].set_yticks([], [])

    return

def rightmost_figure(ax):

    #--> Maillage de l'espace des phases.
    x = np.linspace(-1, 1, 10)
    x, y = np.meshgrid(x, x)

    #--> Définition du système linéarisé
    f = lambda x, y: np.array([-x , 2*x - y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[0], x, y, xdot, ydot, density=0.5)
    ax[0].plot(0, 0, 'o', color=color[1], ms=4)
    ax[0].set_xticks([], [])
    ax[0].set_yticks([], [])
    ax[0].set_title('Noeud impropre', fontsize=14)

    #--> Définition du système linéarisé
    f = lambda x, y: np.array([x, 5*y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[1], x, y, xdot, ydot, density=0.5)
    ax[1].plot(0, 0, 'o', color=color[1], ms=4)
    ax[1].set_xticks([], [])
    ax[1].set_yticks([], [])
    ax[1].set_title('Noeud repulsif', fontsize=14)

    #--> Définition du système linéarisé
    f = lambda x, y: np.array([x, -y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[2], x, y, xdot, ydot, density=0.5)
    ax[2].plot(0, 0, 'o', color=color[1], ms=4)
    ax[2].set_title('Col', fontsize=14)
    ax[2].set_xticks([], [])
    ax[2].set_yticks([], [])

    return

def leftmost_figure(ax):

    #--> Maillage de l'espace des phases.
    x = np.linspace(-1, 1, 10)
    x, y = np.meshgrid(x, x)

    #--> Définition du système linéarisé
    f = lambda x, y: np.array([-x, -5*y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[0], x, y, xdot, ydot, density=0.3)
    ax[0].plot(0, 0, 'o', color=color[1], ms=4)
    ax[0].set_title('Noeud attractif', fontsize=14)
    ax[0].set_xticks([], [])
    ax[0].set_yticks([], [])

    #--> Définition du système linéarisé
    f = lambda x, y: np.array([-x-y, -x-y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[1], x, y, xdot, ydot, density=0.333)
    ax[1].plot(0, 0, 'o', color=color[1], ms=4)
    ax[1].set_xticks([], [])
    ax[1].set_yticks([], [])
    ax[1].set_title("Sur l'axe", fontsize=14)

    #--> Definition du systeme linearise
    f = lambda x, y: np.array([-x+y, -x+y])

    #--> Calcul de dx/dt et dy/dt
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f(x[:], y[:])

    streamlines(ax[2], x, y, xdot, ydot, density=0.333)
    ax[2].plot(0, 0, 'o', color=color[1], ms=4)
    ax[2].set_title("A l'origine", fontsize=14)
    ax[2].set_xticks([], [])
    ax[2].set_yticks([], [])

    return



def summary():

    fig = plt.figure()
    fig.set_size_inches(10, 10)

    gs = GridSpec(5, 5)

    #--> Figure centrale.
    x = np.linspace(-1, 1)
    X, Y = np.meshgrid(x, x)

    x_pos = np.linspace(0, 1)
    x_neg = np.linspace(-1, 0)
    a0 = fig.add_subplot(gs[1:-1, 1:-1])
    a0.fill_between(x, x**2, 0, facecolor='blue', alpha=0.25)
    a0.fill_between(x, x**2, 1, facecolor='orange', alpha=0.25)
    a0.fill_between(x, 0*x, -1, facecolor='green', alpha=0.25)

    c1 = a0.contour(X, Y, X**2.-Y, levels=[-1e-6], colors='k')
    fmt = {}
    strs = [r'$\Delta=0$']
    for l, s in zip(c1.levels, strs):
        fmt[l] = s
    a0.clabel(c1, c1.levels, inline=True, fmt=fmt, fontsize=12)

    a0.plot(x_pos, 0*x_pos, color='k')
    a0.plot(x_neg, 0*x_neg, color='k')
    a0.plot(0*x_pos, x_pos, color='k')

    a0.set_xticks([], [])
    a0.set_yticks([], [])

    a0.set_xlabel(r'$tr({\bf J})$')
    a0.set_ylabel(r'$det({\bf J})$')

    a0.set_ylim(-1, 1)

    #--> Portrait de phase superieurs.
    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[0, 2])
    a3 = fig.add_subplot(gs[0, 3])

    upper_figure([a1, a2, a3])

    #--> Portrait de phase a droite.
    a4 = fig.add_subplot(gs[1, -1])
    a5 = fig.add_subplot(gs[2, -1])
    a6 = fig.add_subplot(gs[3, -1])

    rightmost_figure([a4, a5, a6])

    #--> Portrait de phase a droite.
    a7 = fig.add_subplot(gs[1, 0])
    a8 = fig.add_subplot(gs[2, 0])
    a9 = fig.add_subplot(gs[3, 0])

    leftmost_figure([a7, a8, a9])

    #--> Draw the upper connectors.
    xyA = (.0, -1)
    xyB = (-.25, .75)
    con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data', axesA=a1, axesB=a0, arrowstyle='->', connectionstyle='angle3')
    a1.add_artist(con)

    xyA = (0.5, -1)
    xyB = (0., .66)
    con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data', axesA=a2, axesB=a0, arrowstyle='->', connectionstyle='angle3')
    a2.add_artist(con)

    xyA = (0., -1)
    xyB = (.25, .75)
    con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data', axesA=a3, axesB=a0, arrowstyle='->', connectionstyle='angle3')
    a3.add_artist(con)

    #--> Draw the rightmost connectors.
    xyA = (-1, 0.)
    xyB = (.75, .75**2)
    con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data', axesA=a4, axesB=a0, arrowstyle='->', connectionstyle='arc3')
    a4.add_artist(con)

    xyA = (-1, 0.)
    xyB = (.75, .25)
    con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data', axesA=a5, axesB=a0, arrowstyle='->', connectionstyle='arc3')
    a5.add_artist(con)

    xyA = (-1, 0.)
    xyB = (0., -0.5)
    con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data', axesA=a6, axesB=a0, arrowstyle='->', connectionstyle='arc3')
    a6.add_artist(con)

    #--> Draw the rightmost connectors.
    xyA = (1., -1.)
    xyB = (-.75, .25)
    con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data', axesA=a7, axesB=a0, arrowstyle='->', connectionstyle='arc3')
    a7.add_artist(con)

    xyA = (1., 1.)
    xyB = (-.5, 0.)
    con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data', axesA=a8, axesB=a0, arrowstyle='->', connectionstyle='arc3')
    a8.add_artist(con)

    xyA = (1, 0.)
    xyB = (0., 0.)
    con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data', axesA=a9, axesB=a0, arrowstyle='->', connectionstyle='arc3')
    a9.add_artist(con)

    plt.tight_layout()

    return









def lotka_volterra(alpha=0.):

    #--> Définition du système.
    f = lambda u, v: np.array([u*(1.-v) , alpha*v*(u-1.)])

    #--> Maillage pour l'espace des phases.
    u = np.linspace(0, 3, 100)
    u, v = np.meshgrid(u, u)

    #--> Calcul de du/dt et dv/dt
    udot = np.zeros_like(u)
    vdot = np.zeros_like(v)

    udot[:], vdot[:] = f(u[:], v[:])


    gs = GridSpec(1, 3)
    fig = plt.figure()
    fig.set_size_inches(15, 5)

    #--> Plot le portrait de phase.
    ax0 = fig.add_subplot(gs[0])
    streamlines(ax0, u, v, udot, vdot)
    ax0.set_xlabel(r'Lievres')
    ax0.set_ylabel(r'Lynx')
    ax0.set_aspect('equal')
    ax0.set_title('')
    ax0.locator_params(axis='both', nbins=4)

    #--> Simulation du système.
    def simulation(y, t):
        dy = np.zeros_like(y)
        dy[0] = y[0]*(1-y[1])
        dy[1] = alpha*y[1]*(y[0]-1)
        return dy

    y0 = np.array([0.5, 0.5])
    t = np.linspace(0, 200, 1000)
    sol = odeint(simulation, y0, t)

    ax1 = fig.add_subplot(gs[1:])
    ax1.plot(t, sol[:, 0], label=r'Lievres', color=color[0])
    ax1.plot(t, sol[:, 1], label=r'Lynx', color=color[1])
    ax1.legend(loc=1)
    ax1.set_xlabel(r't')

    plt.tight_layout()

    return





def vanderpol(mu=0.1, beta=0.1):

    #--> Définition du système.
    f = lambda y, t: np.array([y[1] , mu*(1.-beta*y[0]**2)*y[1]-y[0]])

    #--> Création du maillage de l'espace des phases.
    x = np.linspace(-10, 10, 25)
    x, y = np.meshgrid(x, x)

    #--> Calcul de dx/dt et dy/dt.
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    xdot[:], ydot[:] = f([x[:], y[:]], 0)

    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = fig.gca()
    streamlines(ax, x, y, xdot, ydot, density=2)
    ax.set_title('')


    #--> Simulation de quelques trajectoires.
    y0 = np.random.rand(4, 2)
    y0 = y0 - y0.mean()
    y0 *= 10.
    t = np.linspace(0, 100, 1000)
    for y_init in y0:
        ys = odeint(f, y_init, t)
        ax.plot(ys[:, 0], ys[:, 1], 'k')

    return
