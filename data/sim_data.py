import numpy as np
import tensorflow as tf

from p_sampling import user_p_sample
from ggp import sample_ggp
from pmf import pmf

"""
Sample the dataset given parameters
Store the dataset at a given location
Store the true parameters + hyperparameters in a separate file
"""
def sim_data(tu, su, size_u, a, b,
             ti, si, size_i, c, d,
             K, eps):
    # Simulate Data
    with tf.device('/cpu:0'):
        gam = tf.convert_to_tensor(sample_ggp(size_u, su, tu, eps))
        omega = tf.convert_to_tensor(sample_ggp(size_i, si, ti, eps))
        Uc = gam.get_shape().as_list()[0]
        Ic = omega.get_shape().as_list()[0]
        # makes broadcasting later a bit easier
        gam = tf.reshape(gam,[-1,1])
        omega = tf.reshape(omega,[-1,1])
        theta = tf.random_gamma([Uc,K],alpha=a,beta=b)
        beta = tf.random_gamma([Ic,K],alpha=c,beta=d)
        sim_model = pmf(gam, omega, theta, beta, eps)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sample = sim_model._sample_one(sess=sess)

        # remove isolated users by reindexing users so that labels are contiguous
        uids = sample[:,0]
        allusers = np.unique(uids)
        numusers = allusers.size
        convertusers = np.zeros(np.max(allusers)+1, dtype=np.int64)
        convertusers[allusers] = np.r_[0:numusers] # [orig_user_idx] = new_user_idx
        sample[:,0] = convertusers[uids]
        gam_samp = gam.eval()[allusers]
        theta_samp = theta.eval()[allusers,:]

        # remove isolated items
        iids = sample[:,1]
        allitems = np.unique(iids)
        numitems = allitems.size
        convertitems = np.zeros(np.max(allitems)+1, dtype=np.int64)
        convertitems[allitems] = np.r_[0:numitems] # [orig_item_idx] = new_item_idx
        sample[:,1] = convertitems[iids]
        omega_samp = omega.eval()[allitems]
        beta_samp = beta.eval()[allitems,:]


    return gam_samp, theta_samp, omega_samp, beta_samp, sample

