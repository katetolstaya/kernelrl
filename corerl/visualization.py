import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt

def testGrid(X, kernel, resolution=None, count=None, bounds=None):
    '''Generate a misc grid to evaluate kernel functions against.

    Arguments:
      X (Nx2 array) : kernel basis elements
      kernel        : kernel to use
      resolution    : resolution of grid
      count         : number of points in each coordinate axis
      bounds        : 2x2 array giving corners of bounding box to use

    Either 'resolution' or 'count' must be specified, and 'count' will
      take precedence if both are.

    Returns:
    A dictionary with fields:
      X,Y  : the X,Y arrays found as outputs of 'meshgrid' command
      dims : the dimensions of the grid
      Kg   : kernel function of every grid point evaluated against every
             misc point

    To use Kg, it should be dotted with the weight vector, i.e.
      Kg.dot(w) will be a 2-dimensional matrix with each entry corresponding
      to the X,Y coordinate
    '''
    N = len(X)
    xmin, ymin = np.min(X, axis=0) - 0.1
    xmax, ymax = np.max(X, axis=0) + 0.1
    if bounds is not None:
        xmin, ymin = np.min(bounds, axis=0) - 0.1
        xmax, ymax = np.max(bounds, axis=0) + 0.1

    # Create our grid points
    tGrid = {}
    if count is not None:
        tGrid['X'], tGrid['Y'] = np.meshgrid(
                np.linspace(xmin, xmax, count),
                np.linspace(ymin, ymax, count)
                )
    elif resolution is not None:
        tGrid['X'], tGrid['Y'] = np.meshgrid(
                np.arange(xmin, xmax, resolution),
                np.arange(ymin, ymax, resolution)
                )
    else:
        raise "must specify either 'count' or 'resolution'"
    dims = tGrid['X'].shape
    size = tGrid['X'].size
    tGrid['dims'] = dims

    # Evaluate our misc points against every grid point
    tGrid['Kg'] = kernel.f(
            np.hstack((
                tGrid['X'].reshape(size, 1),
                tGrid['Y'].reshape(size, 1)
                )),
            X).reshape(dims[0], dims[1], len(X))

    return tGrid

def plotKernelFunction(tGrid, w, D=None):
    # # Create new figure
    # plt.figure(); plt.hold(True)
    plt.hold(True)
    # Plot the contour
    Z = tGrid['Kg'].dot(w)
    if len(Z.shape) > 2:
        Z = Z.max(axis=2)
    plt.contour(tGrid['X'], tGrid['Y'], Z)
    # Plot the dictionary points
    if D is not None:
        plt.plot(D[:,0], D[:,1], 'k.')

def plotModelLossHistory(histM, histL):
    # Create new figure
    fig = plt.figure()
    # Add loss data on log scale
    ax1 = fig.add_subplot(111)
    line1, = ax1.semilogy(histL,'r')
    plt.ylabel('Testing loss')
    # Add model order
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    line2, = ax2.plot(histM)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    plt.ylabel('Model order')
    # Legend
    plt.xlabel('Training episode')
    plt.legend((line1, line2), ('loss', 'M'))
    return fig

def plot(data):
    config = data.config
    bounds = np.vstack((data.env.observation_space.high, data.env.observation_space.low))
    if isinstance(data.output, list):
        # Plot several experiments
        for (name, approx) in data.output:
            if 'V' in approx:
                resolution = config.getfloat(name, 'PlotResolution', fallback=0.01)
                grid = testGrid(approx.V.D, approx.V.kernel, resolution=resolution, bounds=bounds)
                plotKernelFunction(grid, approx.V.W, approx.V.D)
                plt.title(name)
        # Plot loss curves
        names  = [t[0]          for t in data.output if 'historyL' in t[1]]
        curves = [t[1].historyL for t in data.output if 'historyL' in t[1]]
        if curves:
            plt.figure(); plt.hold(True)
            for curve in curves:
                plt.semilogy(curve)
            plt.title('Testing Loss')
            plt.xlabel('Training episode')
            plt.legend(names)
        # Plot model order curves
        names  = [t[0]          for t in data.output if 'historyM' in t[1]]
        curves = [t[1].historyM for t in data.output if 'historyM' in t[1]]
        if curves:
            plt.figure(); plt.hold(True)
            for curve in curves:
                plt.plot(curve)
            plt.title('Model Order')
            plt.xlabel('Training episode')
            plt.legend(names)
        # Plot cumulative reward curves
        names  = [t[0]          for t in data.output if 'historyR' in t[1]]
        curves = [t[1].historyM for t in data.output if 'historyR' in t[1]]
        if curves:
            plt.figure(); plt.hold(True)
            for curve in curves:
                plt.plot(curve)
            plt.title('Cumulative (Testing) Reward')
            plt.xlabel('Training episode')
            plt.legend(names)
    else:
        approx = data.output
        if 'V' in approx:
            resolution = config.defaults().get('plotresolution', 0.01)
            grid = testGrid(approx.V.D, approx.V.kernel, resolution=resolution, bounds=bounds)
            plotKernelFunction(grid, approx.V.W, approx.V.D)
        if 'Q' in approx:
            resolution = config.defaults().get('plotresolution', 0.01)
            grid = testGrid(approx.Q.D, approx.Q.kernel, resolution=resolution, bounds=bounds)
            plotKernelFunction(grid, approx.Q.W, approx.Q.D)
        # Plot loss curve
        if 'historyL' in approx:
            plt.figure()
            plt.semilogy(approx.historyL)
            plt.title('Testing Loss')
            plt.xlabel('Training episode')
        # Plot model order curve
        if 'historyM' in approx:
            plt.figure()
            plt.plot(approx.historyM)
            plt.title('Model Order')
            plt.xlabel('Training episode')
        # Plot model order curve
        if 'historyR' in approx:
            plt.figure()
            plt.plot(approx.historyR)
            plt.title('Cumulative (Testing) Reward')
            plt.xlabel('Training episode')
    plt.show(False)

def plot_metrics(data):
    # Build a list of all metrics
    metricnames = set()
    for logs in data.itervalues():
        metricnames.update(set(logs['interval_metrics'].keys()))
    metricnames.remove('step')
    # Build plots of all metrics
    for metric in metricnames:
        names = []
        plt.figure(metric); plt.clf(); plt.hold(True)
        for experiment, logs in data.iteritems():
            if metric in logs['interval_metrics']:
                names.append(experiment)
                plt.plot(logs['interval_metrics']['step'], logs['interval_metrics'][metric])
        plt.title(metric); plt.xlabel('Training Step'); plt.ylabel(metric); plt.legend(names)

import time
def animate(data):
    config = data.config[data.config.default_section]
    T      = config.getint('MaximumEpisodeLength')
    approx = data.output
    env = data.env
    delay=1.0/env.metadata.get('video.frames_per_second', 30)
    s = env.reset()
    env.render()
    for i in range(T):
        Qs = approx.Q(s)
        a = nr.choice(np.argwhere(Qs == Qs.max()).flatten())
        s, r, done, _ = env.step(a)
        env.render()
        if done:
            print ('terminated after %s timesteps' % i)
            break
        time.sleep(delay)

