import multiprocessing as mp

import numpy as np
from scipy.stats import linregress

from p_sampling import user_p_sample
from data.sim_data import sim_data


def rho_after_p_samp(graph, p):
    """
    user_p_sample a graph and compute rho (graph density) 
    rho = |E|/(|V_I|*|V_U|)
    """
    samp,_ = user_p_sample(graph,p)
    U = np.unique(samp[:,0]).shape[0]
    I = np.unique(samp[:,1]).shape[0]
    E = samp.shape[0]
    return np.float(E)/(U*I), U, I


def user_p_samp_stats(graph, n_samp = 100):
    """
    Returns |V_U|, |V_I| and |E| after repeated p-sampling
    """
    items = np.zeros(n_samp)
    users = np.zeros(n_samp)
    occ_pairs = np.zeros(n_samp)

    p_incr = 1. / n_samp
    samp = np.copy(graph)
    p_last = 1.
    for i in range(n_samp):
        p_target = 1. - i*p_incr # 'p' value for ith entry
        p = p_target / p_last # a p-sampling of samp(G, p_last) is samp(G, p*p_last)

        samp, _ = user_p_sample(samp, p)

        users[i] = np.unique(samp[:, 0]).shape[0]
        items[i] = np.unique(samp[:, 1]).shape[0]
        occ_pairs[i] = samp.shape[0]

        p_last = p_target

    return users, items, occ_pairs


def est_si(graph, p_base=0.9):
    """
    heuristic for setting sigma. Based on the idea that asymptotically
    log(items) ~ c + log(alpha) + si*log(s) and sim for items

    :param graph:
    :return:
    """
    n_samp = 10

    items = np.zeros(n_samp)
    samp = np.copy(graph)

    for i in range(n_samp):
        _, samp = user_p_sample(samp, 1.-p_base) # logically equivalent to samp, _ = user_p_sample(samp, p), but mb faster
        items[i] = np.unique(samp[:, 1]).shape[0]


    slope = linregress(x=np.arange(1,n_samp+1), y=np.log(items)).slope
    #estimate:

    # I = C + si * log(s) and s=p^n * s_0 implies
    # I = C' + si * log(p) * n ; i.e. si = slope / log(p)
    return slope / np.log(p_base)



# \hat(su)[p] = -(log(U_s) - log( \sum_{users} 1 - (1-p)^-D_u )) / log(p)
# 
def estimate_sigmas_det(graph, N=100):
    """
    Deterministic estimator for Sigma. 
    In this version we numerically integrate out p from the deterministic estimator.
    """
    su = np.zeros(N)
    si = np.zeros(N)
    U, Du = np.unique(graph[:,0], return_counts=True)
    I, Di = np.unique(graph[:,1], return_counts=True)
    U = U.shape[0]
    I = I.shape[0]
    p = np.linspace(0.001,0.999,N)
    for i in range(N-1):
        # compute degrees
        # su = -(np.log(U) - np.log(np.sum(1-np.power(1-p_base, Du)))) / np.log(p_base)
        # si = -(np.log(I) - np.log(np.sum(1-np.power(1-p_base, Di)))) / np.log(p_base)
        su[i] = np.log(1 - np.sum(np.power(1-p[i], Du))/U) / np.log(p[i])
        si[i] = -(np.log(I) - np.log(np.sum(1-np.power(1-p[i], Di)))) / np.log(p[i])
    return [float(np.mean(si)), float(np.mean(su))]

def estimate_sigmas(graph, p_base=0.9):
    """
    estimate for su, si used to generate graph. This works when su, si > 0 (i.e. sparse in both users and items)
    if either su < 0 or si < 0 then estimate will be appx 0, but this should be diagnosed by examining size plots

    :param graph:
    :return: si_est, su_est both real numbers in [0,1)
    """

    # interchange items and users and use same est function (based on user_p_sampling)
    g_flip = graph.copy()
    g_flip[:,0] = graph[:,1]
    g_flip[:,1] = graph[:,0]

    return [est_si(graph, p_base), est_si(g_flip, p_base)]

def estimate_size(graph,
                  tu, su, a, b,
                  ti, si, c, d,
                  K):
    """
    estimate item size and user size of graph, given the hyperparameters
    Main idea: U ~ C * s * alpha^su; edges ~ C' * alpha * s
    so log(U) - su*log(e) ~ C'' + (1-su)*log(s)
    and sim
    log(I) - si*log(e) ~ C''' + (1-si) * log(alpha)
    We can either use Zach's analytic expressions for the constants or
    (as we do here) compute estimates via simulation
    """

    U = np.unique(graph[:,0]).shape[0]
    I = np.unique(graph[:,1]).shape[0]
    op = graph.shape[0] # occupied pairs
    N = 5
    sim_size_u = 100.
    sim_size_i = 100.
    eps = 1e-8
    item_size = np.zeros(N)
    user_size = np.zeros(N)
    for n in range(N):    
        _, _, _, _, sim = sim_data(tu, su, sim_size_u, a, b,
                                   ti, si, sim_size_i, c, d,
                                   K, eps)
        sim_U = np.unique(sim[:,0]).shape[0]
        sim_I = np.unique(sim[:,1]).shape[0]
        sim_op = sim.shape[0] # occupied pairs

        if su > 0:
            # idea: log(U) - su*log(e) ~ C'' + (1-su)*log(s)
            # so log(U/sim_U) - su*log(edges/sim_edges) ~ (1-su) * log(s/sim_user_size)
            # and similarly for s
            log_user_size = (np.log(np.float(U) / sim_U) - su * np.log(np.float(op) / sim_op)) / (1.-su) + np.log(sim_size_u)
            user_size[n] = np.exp(log_user_size)
        else:
            # number of GGP atoms ~ C * user_size
            # estimate size by assuming every ggp item shows up as a vertex
            user_size[n] = sim_size_u * U / sim_U

        if si > 0:
            log_item_size = (np.log(np.float(I) / sim_I) - si * np.log(np.float(op) / sim_op)) / (1.-si) + np.log(sim_size_i)
            item_size[n] = np.exp(log_item_size)
        else:
            # number of GGP atoms ~ C * user_size
            # estimate size by assuming every ggp item shows up as a vertex
            item_size[n] = sim_size_i * I / sim_I

    return np.mean(item_size), np.mean(user_size)

def appx_edge_scaling(tu, su, a, b,
                      ti, si, c, d,
                      K):
    """
    graph size goes as (occupied_pairs)_alpha,s = C(hyperparms) * alpha*s
    this returns an estimate for C
    """

    size_u = 100.
    size_i = 100.
    eps = 1e-8

    _, _, _, _, edges = sim_data(tu, su, size_u, a, b,
             ti, si, size_i, c, d,
             K, eps)

    return np.float(edges.shape[0]) / (size_u*size_i)
