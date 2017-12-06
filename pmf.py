import tensorflow as tf
import numpy as np

from tensorflow.contrib.distributions import Distribution, NOT_REPARAMETERIZED
from tensorflow.contrib.distributions import Poisson


class pmf(Distribution):
    def __init__(self,
                 gam, omega, theta, beta,
                 eps,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="pmf",
                 value=None
                 ):
        """
        Poisson matrix factorization

        gam, omega are approximate; epsilon is approximation level used in the simulation

        """

        self.K = theta.shape.as_list()[1]

        parameters = locals()

        with tf.name_scope(name, values=[gam,omega,theta,beta]) as ns:
            # TODO: does this screw up composition?
            with tf.control_dependencies([]):
                self._gam = tf.identity(gam, name="xi")
                self._omega = tf.identity(omega, name="eta")
                self._theta = tf.identity(theta, name="theta")
                self._beta = tf.identity(beta, name="beta")

        super(pmf, self).__init__(
            dtype=tf.float32,
            parameters=parameters,
            # is_continuous=False,
            reparameterization_type=NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns)

        # TODO: I put this in to mimic the behaviour of the tf distributions, but I dunno...
        self._kwargs = {"gam": self._gam,
                        "omega": self._omega,
                        "theta": self._theta,
                        "beta": self._beta}

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

    def _sample_one_v1(self, seed=None):
        """
        Roughly, draw a single sample from the model.
        Actually, we return a matrix of size Uc x Ic that includes 0 degree vertices, as well as the indices
        of the users and items with degree > 0
        The reason is that it makes it easier to track the 'ground truth' parameters for estimation testing

        Note that making this broadcast like other tf dists will be painful, because sample shape is not fixed
        :param seed:
        :return:
        """
        gam = self._gam
        omega = self._omega
        theta = self._theta
        beta = self._beta

        user_weights = theta*gam
        item_weights = beta*omega
        edge_weight_expct = tf.matmul(user_weights,item_weights, transpose_b=True)

        def np_one_sample(lam, seed=None):
            prng = np.random.RandomState(seed)
            A = prng.poisson(lam)
            user_indices = np.where(np.sum(A > 0, 1))[0]
            item_indices = np.where(np.sum(A > 0, 0))[0]
            return user_indices.astype(np.int64), item_indices.astype(np.int64), A.astype(np.float32)

        user_indices, item_indices, sample = \
            tf.py_func(np_one_sample, [edge_weight_expct], [tf.int64, tf.int64, tf.float32])

        # valid since we haven't dropped the 0 degree verts
        sample.set_shape(edge_weight_expct.get_shape())

        return user_indices, item_indices, sample

    def _sample_one(self, sess, seed=None):
        """
        Roughly, draw a single sample from the model.

        Warning: this doesn't really behave like a normal tensorflow distribution... it's deceptive as is
        (Because I want to use some numpy functionality, and I didn't want to bother w wrappers)

        The return is an array of the form [user_idx, item_idx, edge_multiplicity]
        where user_idx and item_idx are into gam and omega respectively

        :param sess: tf session
        :param seed:
        :return:
        """
        with tf.device("/cpu:0"):
            gam = self._gam
            omega = self._omega
            theta = self._theta
            beta = self._beta

            user_weights = theta*gam
            item_weights = beta*omega

            i_mass_tots = tf.reduce_sum(item_weights,0) # total mass of each type in items
            u_mass_tots = tf.reduce_sum(user_weights,0)

            # K probability distributions over items / users
            i_logits = tf.expand_dims(tf.log(tf.transpose(item_weights)),0)
            u_logits = tf.expand_dims(tf.log(tf.transpose(user_weights)),0)

            # total number of edges of each type
            tot_edges_mean = u_mass_tots * i_mass_tots
            tot_edges = tf.cast(tf.random_poisson(tot_edges_mean,[1])[0], tf.int32)

            # assign edges to pairs
            item_assignments = [tf.multinomial(i_logits[:,k,:],tot_edges[k]) for k in range(self.K)]
            user_assignments = [tf.multinomial(u_logits[:,k,:],tot_edges[k]) for k in range(self.K)]

            edge_list = tf.concat([tf.squeeze(tf.stack([user_assignments[k], item_assignments[k]])) for k in range(self.K)], axis=1)

            # we now actually run this so we can use some functionality in np.sort that doesn't exist for tf.sort
            redundant_edge_list = sess.run(edge_list)
            # print("redundant edge list done")

            uniques = np.unique(redundant_edge_list, return_counts=True, axis=1)
            # print("edge list done")

        return np.vstack([uniques[0], np.expand_dims(uniques[1], 0)]).T

    def _log_prob(self, As):
        raise NotImplementedError("log_prob is not implemented")
        # Note: this is a little bit tricky because we need to specificy said which elements of A correspond to which atoms


def _np_one_sample(lam, seed=None):
    # TODO: need to modify this so we can keep track of which vertices stay in the graph (otherwise hosed)
    # TODO: only required for very large samples, so this is currently left unused
    """
    Samples according to:
    1. A[i,j] ~ poi(lam[i,j]) for all i,j
    2. discard any all 0 rows and all 0 columns

    Note: np.random.poisson(lam) has memory issues, hence our generator approach
    :param lam: np.array, float
    :param seed: int, seed for rng
    :return: numpy array, cast to np.float32 for convenience with tensorflow
    """

    assert lam.ndim == 2

    def big_poi_mat_gen(lam, prng):
        # sample row by row, keeping each row only if it has at least one non-zero entry
        for i in range(lam.shape[0]):
            row_poi = prng.poisson(lam[i,:])
            if np.sum(row_poi)>0:
                yield row_poi

    prng = np.random.RandomState(seed)
    no_empty_rows = np.array([poi_row for poi_row in big_poi_mat_gen(lam, prng)])
    # we now need to drop the empty columns
    no_empty_rows_or_cols=no_empty_rows[:, np.where(np.sum(no_empty_rows > 0, 0))[0]]
    return no_empty_rows_or_cols.astype(np.float32)