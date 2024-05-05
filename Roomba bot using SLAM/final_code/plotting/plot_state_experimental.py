import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
np.set_printoptions(precision=6)
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)

def plot_state(state_history, covariance_history, id_history, ar=0.2, trajectory = None):
    if len(state_history) > 0:
        latest_state = state_history[-1]
        # latest_covariance = covariance_history[-1]
        x = latest_state.reshape(-1,3)
        # P = latest_covariance
        ids = id_history[-1]
        ax.clear()
        # print('x:', x.shape, x)
        # print('P:', P.shape, P)
        # print('lm_ids:', ids)

        for idx, lm_id in enumerate(ids):
            lm_x = x[idx+1]
            # print(lm_x)
            # lm_P_2D = P[3*idx+3:3*idx+5, 3*idx+3:3*idx+5]
            # print('lm_x {}:'.format(lm_id), lm_x)
            # print('lm_P_2D {}:'.format(lm_id), lm_P_2D)
            
            ax.text(lm_x[0], lm_x[1], 'LM{}'.format(lm_id))
            # draw_ellipse(lm_x, lm_P_2D, ax, n_std=1.0, edgecolor='b')
            ax.arrow(lm_x[0], lm_x[1], ar*np.cos(lm_x[2]), ar*np.sin(lm_x[2]))
            ax.arrow(lm_x[0], lm_x[1], -ar*np.cos(lm_x[2]), -ar*np.sin(lm_x[2]))
            ax.arrow(lm_x[0], lm_x[1], ar/2*np.cos(lm_x[2]+np.pi/2), ar/2*np.sin(lm_x[2]+np.pi/2))

        if trajectory is not None:
            trajectory[0].append(x[0,0])
            trajectory[1].append(x[0,1])
            ax.plot(trajectory[0], trajectory[1])
        
        ax.scatter(x[:,0], x[:,1])
        ax.arrow(x[0,0], x[0,1], ar*np.cos(x[0,2]), ar*np.sin(x[0,2]))
        ax.text(x[0,0], x[0,1], 'Robot')
        # draw_ellipse(x[0], P[0:2,0:2], ax, n_std=1.0, edgecolor='r')

        # if 'x_pred' in latest_state:
        #     x_pred = latest_state['x_pred']
        #     ax.scatter(x_pred[0,0], x_pred[1,0], c='r')
        #     ax.arrow(x_pred[0,0], x_pred[1,0], ar*np.cos(x[0,2]), ar*np.sin(x[0,2]), color='r')
        
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.axis('equal')
        ax.set_xlim((-1.5, 2.5))
        ax.set_ylim((-1.5, 2.5))
        ax.set_title('map at time: {}'.format(len(state_history)))

        plt.savefig("/home/ramia/ucsd/cse276a/hw5/plots/plot-{:04d}.jpg".format(len(state_history)))
        
        # plt.pause(0.0005)


def draw_ellipse(x, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Ref: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x : array-like, shape (2), position.
    P : array-like, shape (2,2), covariance.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/(np.sqrt(cov[0, 0] * cov[1, 1]) + 1e-8)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Use the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x[0], x[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Ref: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_pickle():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        file_path = sys.argv[1]
    else:
        rosws = "/home/ramia/ucsd/cse276a/hw5/"
        state_file_path = os.path.join(rosws, 'state_history_firstrun.pkl')
        # cov_file_path = os.path.join(rosws, 'covariance_history_firstrun.pkl')
        id_file_path = os.path.join(rosws, 'id_history_firstrun.pkl')
    with open(state_file_path, 'rb') as fp:
        state_history = pickle.load(fp, encoding="latin1")
    # with open(cov_file_path, 'rb') as fp:
    #     covariance_history = pickle.load(fp)
    with open(id_file_path, 'rb') as fp:
        id_history = pickle.load(fp, encoding="latin1")
    trajectory = [[], []]

    for i in tqdm(range(0, len(state_history), 5)):
        # plot_state(state_history[0:i], covariance_history[0:i], id_history[0:i], trajectory = trajectory)
        plot_state(state_history[0:i], None, id_history[0:i], trajectory = trajectory)
    
    # plt.show()


if __name__ == "__main__":
    plot_pickle()
