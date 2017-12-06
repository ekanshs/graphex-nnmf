"""
Truncated Poisson distribution.
Z ~ Poi(lam)
X ~ Z | Z>0

TBD: write this as a tf.contrib.Distribution

Idea is to use this as part of the variational scheme in the setting where multigraph is reported as a simple graph
"""

import tensorflow as tf
import numpy as np
from helpers import to_prob

from tensorflow.contrib.distributions import Distribution, NOT_REPARAMETERIZED
from tensorflow.contrib.distributions import Poisson
from tensorflow.python.framework import ops


class tPoisson(Distribution):
    def __init__(self,
                 lam,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="tPoisson",
                 value=None
                 ):
        """
        Truncated Poisson distribution (Poisson conditional on > 0)
        """

        parameters = locals()

        with tf.name_scope(name, values=[lam]) as ns:
            with tf.control_dependencies([]):
                self._lam = ops.convert_to_tensor(lam, name="lam")

        super(tPoisson, self).__init__(
            dtype=tf.float32,
            parameters=parameters,
            reparameterization_type=NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns)

        # TODO: I put this in to mimic the behaviour of the tf distributions, but I dunno...
        self._kwargs = {"lam": self._lam}

    # TODO: properties and shapes
    # @property
    # def zs(self):
    #     """Community ids."""
    #     return self._zs
    #
    # @property
    # def eta(self):
    #     """Link weights"""
    #     return self._eta
    #
    # @property
    # def n_comm(self):
    #     """number of communities"""
    #     return self._n_comm

    # def _batch_shape(self):
    #     return tf.convert_to_tensor(self.get_batch_shape())

    # def _get_batch_shape(self):
    #     # TODO: not sure about this

    # def _event_shape(self):
    #     return tf.convert_to_tensor(self.get_event_shape())

    # def _get_event_shape(self):
    #     # U x I... but not determined ahead of time

    def mean(self):
        lam = self._lam
        nz_ind = tf.greater(lam, 0.0001) # if lam is too small then the mean computes as 0/0
        mean = lam / (1.-tf.exp(-lam))
        return tf.where(nz_ind, mean, tf.ones_like(mean))


class tPoissonMulti(Distribution):
    def __init__(self,
                 log_lams,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="tPoissonMulti",
                 value=None
                 ):
        """
        Truncated Poisson distribution (Poisson conditional on > 0)
        Parameterized by (lam_1, .., lam_K) (the natural parameters of the exponential family are log(lam_k))

        :param log_lams: d_1 x d_2 x ... x d_n x K tensor with float type, where K is parameter dimension
        :param validate_args:
        :param allow_nan_stats:
        :param name:
        :param value:
        """

        parameters = locals()

        with tf.name_scope(name, values=[log_lams]) as ns:
            with tf.control_dependencies([]):
                self.log_lams = ops.convert_to_tensor(log_lams, name="log_lams")

        super(tPoissonMulti, self).__init__(
            dtype=tf.float32,
            parameters=parameters,
            reparameterization_type=NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns)

        self._kwargs = {"log_lams": self.log_lams}

    # TODO: properties and shapes

    def mean(self):
        lams = tf.exp(tf.clip_by_value(self.log_lams,-100.,50.))

        lam_tots = tf.reduce_sum(lams, -1, keep_dims=True)
        # total expected number of edges, use taylor expansion for small entries to dodge numerical stability issues
        total_expct = tf.where(tf.greater(lam_tots, 1e-3),
                               lam_tots/(1.-tf.exp(-lam_tots)),
                               1. + lam_tots/2.)

        return total_expct * to_prob(lams)