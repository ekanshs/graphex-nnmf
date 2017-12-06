import tensorflow as tf
import numpy as np
TINY = 1e-7
from helpers import gather_along_second_axis, gather_and_add


def user_updates(q_gam, q_theta, q_omega, q_beta, q_w, qm_du, a, b, su, tu, size, comp_rem=True, n_samp=95):
    """
    Compute updated parameters for q_gam, q_theta, and q_g
    Note for later use that \natgrad_lam L = lam_new - lam
    :param qm_du[u,k]: tensor; the expected weighted degree of user u counting only edges of type k.
        qm_du[u,k] = E_q[e^k_i.] in the language of the paper
    :param comp_rem: bool, if True compute an estimate of the remainder mass for each of the K user categories,
    otherwise return tf.zeros([K,1]). Computing remainders is slow and shouldn't affect inference much.
    """

    # dim [K,1]; this quantity is used for all user updates so we just compute it once
    ii_tot_mass_m = tf.matmul(q_beta.mean(), q_omega.mean(), transpose_a=True) # expected total mass of instantiated items
    i_tot_mass_m = ii_tot_mass_m + q_w.mean() # expected total mass of items

    # compute the effective expected degrees
    qm_degs_u = tf.reduce_sum(qm_du, 1, keep_dims=True)

    ### gamma param updates
    new_gam_shp = -su + qm_degs_u
    new_gam_rte = tu + tf.matmul(q_theta.mean(), i_tot_mass_m)

    ### theta param updates
    new_theta_shp = a + qm_du
    new_theta_rte = b + q_gam.mean()*tf.transpose(i_tot_mass_m)

    with tf.name_scope("comp_rem_mass"):
        K = q_theta.batch_shape.as_list()[1]
        if comp_rem:
            new_g = size*appx_remain_mass_mean_rate(n_samp, q_omega, q_beta, ii_tot_mass_m, q_w, su, tu, a, b, K)
        else:
            new_g = tf.zeros([K,1])

    return new_gam_shp, new_gam_rte, new_theta_shp, new_theta_rte, new_g


def appx_remain_mass_mean_rate(n_samp, q_omega, q_beta, ii_tot_mass_m, q_w, su, tu, a, b, K):
    """
    Monte carlo approximation for expectation of the leftover mass rate.
    E[leftover mass] = (mass_mean_rate)*size

    """
    theta_rem_samp = tf.random_gamma([n_samp,K], a, b)

    # we also need to draw samples from the total item mass
    # this is potentially quite expensive, so we just approximate the sum as a Gamma

    #compute the variance of the total item mass
    ii_tot_mass_v = tf.reduce_sum(q_beta.variance() * q_omega.variance() +
                                  q_beta.variance() * tf.square(q_omega.mean()) +
                                  tf.square(q_beta.mean()) * q_omega.variance(), axis=0, keep_dims=True)
    ii_tot_mass_v = tf.transpose(ii_tot_mass_v)

    # params for gamma dist approximation
    ii_tot_mass_beta = ii_tot_mass_m / ii_tot_mass_v
    ii_tot_mass_alpha = ii_tot_mass_m * ii_tot_mass_beta

    # sample total item mass
    i_tot_mass_sample = tf.squeeze(tf.random_gamma([n_samp], alpha=ii_tot_mass_alpha, beta=ii_tot_mass_beta)
                                   + q_w.sample([n_samp]))

    # compute the estimate
    denom = tf.pow(tu + tf.reduce_sum(theta_rem_samp*i_tot_mass_sample, axis=1, keep_dims=True), 1-su)
    leftover_samps = theta_rem_samp / denom
    estimated_expect = tf.reduce_mean(leftover_samps, 0, keep_dims=True) # monte carlo average

    # can also compute variance... but it'll turn out to be really tiny
    # var_samps = (1.-su) * tf.square(theta_rem_samp) / tf.pow(tu + tf.reduce_sum(theta_rem_samp*i_tot_mass_sample, axis=1, keep_dims=True), 2-su)
    # estimated_var = tf.reduce_mean(var_samps, 0, keep_dims=True)

    return tf.transpose(estimated_expect)


def multi_graph_edge_update(q_theta, q_beta, es_ind):
    """
    For occupied pair (i,j) w index m the update is:
    lphi[m,:] <- \EE[log(theta[i,:])] + \EE[log(beta[j,:])]

    Peak memory cost for this operation is approximately edges * K * (1 + 1/num_splits)
    Compute time scales linearly with num_splits
    """

    # E[log(X)]
    ltheta = tf.digamma(q_theta.concentration)-tf.log(q_theta.rate)
    lbeta = tf.digamma(q_beta.concentration)-tf.log(q_beta.rate)

    # for occupied pair (i,j) w index m we have oc_theta[m]=ltheta[i,:]
    oc_theta = tf.gather(ltheta, es_ind[:,0])
    oc_beta = tf.gather(lbeta, es_ind[:,1])
    edge_params = oc_theta + oc_beta

    return edge_params


def simple_graph_edge_update(q_theta, q_beta, q_gam, q_omega, es_ind):
    """
    For occupied pair (i,j) w index m the update is:
    edge_param[i,j,:] <- E[log(theta[i,:])] + E[log(beta[j,:])] + E[log(gam[i])] + E[log(omega[i])]
    where edge_param is log(lam), the natural parameters of the truncated Poisson distribution

    Peak memory cost for this operation is approximately edges * K * (1 + 1/num_splits)
    Compute time scales linearly with num_splits
    """

    # E[log(X)]
    ltheta = tf.digamma(q_theta.concentration) - tf.log(q_theta.rate)
    lbeta = tf.digamma(q_beta.concentration) - tf.log(q_beta.rate)
    lgam = tf.digamma(q_gam.concentration) - tf.log(q_gam.rate)
    lomega = tf.digamma(q_omega.concentration) - tf.log(q_omega.rate)

    user_params = lgam + ltheta
    item_params = lomega + lbeta

    # for occupied pair (i,j) w index m we have oc_theta[m]=ltheta[i,:]
    oc_user_params = tf.gather(user_params, es_ind[:,0])
    oc_item_params = tf.gather(item_params, es_ind[:,1])
    edge_params = oc_user_params + oc_item_params

    return edge_params
