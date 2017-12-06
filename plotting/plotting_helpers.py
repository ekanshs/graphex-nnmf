import numpy as np
import matplotlib.pyplot as plt
from helpers import to_prob
from p_sampling import user_p_sample

def plot_deg_counts(edge_list, log_scale=True, xlabel="User Degree", ylabel="Fraction of Users", ax=None, color='blue', fontsize=14, title=None, label=''):
    """
    plot degree distribution of *simple* graph described by edge_list
    :param edge_list:
    :return:
    """
    
    # the number of occurances of each user in the edge list is the same as the user's degree
    samp_users, samp_degs = np.unique(edge_list[:,0], return_counts=True)
    deg, num_user_w_deg = np.unique(samp_degs, return_counts=True)

    # if ymax is not None: plt.ylim([0.8,ymax])
    # if xmax is not None: plt.xlim([0.8,xmax])
    if ax:
        if title:
            ax.set_title(title, fontsize=fontsize+4, fontweight='heavy')

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        # return plt.hist(deg, weights=num_user_w_deg, bins=range(2000))
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)

        ax.plot(deg, num_user_w_deg / np.sum(num_user_w_deg).astype(np.float),"o", color=color,label=label)
    else:
        if title:
            plt.title(title, fontsize=fontsize+2, fontweight='heavy')

        if log_scale:
            plt.xscale('log')
            plt.yscale('log')

        # return plt.hist(deg, weights=num_user_w_deg, bins=range(2000))
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)

        plt.plot(deg, num_user_w_deg / np.sum(num_user_w_deg).astype(np.float),"o", color=color,label=label)


def plot_deg_counts_users(edge_list, log_scale=True, ax=None,color='blue',fontsize=14, title=None, label=''):
    """
    plot degree distribution of *simple* graph described by edge_list
    plot_deg_counts is a nisnomer for plot_deg_counts_users. 
    This function clarifies what exactly gets plotted
    :param edge_list:
    :return:
    """
    plot_deg_counts(edge_list, log_scale=True, xlabel="User Degree",ylabel="Fraction of Users", ax=ax, color=color, fontsize=fontsize, title=title, label=label)

def plot_deg_counts_items(edge_list, log_scale=True, ax=None, color='blue', fontsize=14, title=None,label=''):
    """
    plot degree distribution of *simple* graph described by edge_list
    :param edge_list:
    :return:
    """
    plot_deg_counts(edge_list[:,[1,0,2]], log_scale, xlabel="Item Degree", ylabel="Fraction of Items", ax=ax, color=color, fontsize=fontsize, title=title,label=label)


def plot_rho_v_psamp_users(edges, p_base=0.9, n_samp=20, N=1, title="", color='blue'):
    def rho(edges):
        U = np.unique(edges[:,0]).shape[0]
        I = np.unique(edges[:,1]).shape[0]
        rho = np.float(edges.shape[0])/np.float(U*I)
        return rho

    P = np.power(p_base,range(n_samp))
    rho_matrix = np.zeros((N, n_samp))
    rho_matrix[:,0] = rho(edges)
    for i in range(N):
        samp = np.copy(edges)
        for j in range(n_samp-1):
            _, samp = user_p_sample(samp, 1.-p_base) # logically equivalent to samp, _ = user_p_sample(samp, p), but mb faster
            rho_matrix[i, j+1] = rho(samp)

    # plt.title(title)
    # plt.xlabel("")
    # plt.ylabel("")
    line = plt.plot(P, rho_matrix[0,:],'o',color=color, label=r'User $p$-sampling')
    for i in range(N-1):
        plt.plot(P, rho_matrix[i+1,:],'o',color=color)
    plt.legend()
    return line


def plot_rho_v_psamp_items(edges, p_base=0.9, n_samp=20, N=1, title="", color='green'):
    def rho(edges):
        U = np.unique(edges[:,0]).shape[0]
        I = np.unique(edges[:,1]).shape[0]
        rho = np.float(edges.shape[0])/np.float(U*I)
        return rho
    edges = edges[:,[1,0,2]].copy()
    P = np.power(p_base,range(n_samp))
    rho_matrix = np.zeros((N, n_samp))
    rho_matrix[:,0] = rho(edges)
    for i in range(N):
        samp = np.copy(edges)
        for j in range(n_samp-1):
            _, samp = user_p_sample(samp, 1.-p_base) # logically equivalent to samp, _ = user_p_sample(samp, p), but mb faster
            rho_matrix[i, j+1] = rho(samp)

    # plt.title(title)
    # plt.xlabel("")
    # plt.ylabel("")
    line = plt.plot(P, rho_matrix[0,:],'o',color=color, label=r'Item $q$-sampling')
    plt.legend()
    for i in range(N-1):
        plt.plot(P, rho_matrix[i+1,:],'o',color=color)
    return line
    