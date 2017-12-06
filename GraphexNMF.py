"""
Graphex poisson matrix factorization.
Right now this is mostly to clearly structure our thinking

Python 2.7.9 | Anaconda custom (64-bit)
"""

import numpy as np
import tensorflow as tf
import time

from tensorflow.contrib.distributions import Gamma, Multinomial, softplus_inverse
from helpers import to_prob, compute_degrees2, assign_list, tensor_split, setdiffRows
from coord_updates import user_updates, multi_graph_edge_update, simple_graph_edge_update, appx_remain_mass_mean_rate
from ggp import sample_ggp
from hyperparam_ests import estimate_size
from p_sampling import user_p_sample, item_p_sample
from helpers import zero_index
from dists.tPoisson import tPoissonMulti
from dists.PointMass import PointMass
import pandas as pd

TINY = 1e-7
DEBUG = False

class GraphexNMF(object):

    def __init__(self, edge_idx, edge_vals, U, I,  K, hparams, ground_truth=None, simple_graph=False, GPU=False,
                 fix_item_params=False, comp_rem=True, edge_param_splits=1, seed=None, sess=None, device='/cpu:0',
                 ppm=False):
        """
        Model for Sparse Exchangeable bipartite graph
        
        """

        self.ppm = ppm
        # Launch the session
        if sess:
            self.sess = sess
        else:
            if GPU:
                # For GPU mode
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                config.gpu_options.allocator_type = 'BFC'
                self.sess = tf.Session(config=config)
            else:
                config = tf.ConfigProto(allow_soft_placement=True)
                self.sess = tf.Session(config=config)

        self.device = device
        self.comp_rem = comp_rem
        self.seed = seed
        self.K = K
        self.ground_truth = ground_truth
        self.simple_graph = simple_graph
        self.U, self.I = U, I
        self.fix_item_params = fix_item_params
        self.hparams = hparams
        self.edge_param_splits = edge_param_splits # Splitting GPU parameters to fit according to GPU size
        self.GPU = GPU

        # store the data here:
        self.edge_idx_d = edge_idx

        if self.simple_graph:
            self.edge_vals_d = np.ones(edge_vals.shape[0], dtype=np.float32)
        else:
            self.edge_vals_d = edge_vals.astype(np.float32)

        # create placeholders for the computational graph
        with tf.name_scope("placeholders"):
            with tf.device(self.device):
                self.edge_idx = tf.placeholder(dtype=tf.int32,shape=(edge_idx.shape[0], edge_idx.shape[1]))
                self.edge_vals = tf.placeholder(dtype=tf.float32,shape=(edge_idx.shape[0]))

        if simple_graph:
            # Degree computation without tensorflow. Only works for simple graphs
            _,self.user_degree = np.unique(self.edge_idx_d[:,0],return_counts=True)
            _,self.item_degree = np.unique(self.edge_idx_d[:,1],return_counts=True)
            self.user_degree = self.user_degree.astype(np.float32)
            self.item_degree = self.item_degree.astype(np.float32)
        else:
            with tf.name_scope("init_deg_comp"):
                with tf.device(self.device):
                    user_degree, item_degree = compute_degrees2(tf.expand_dims(self.edge_vals, axis=1), self.edge_idx,
                                                                self.U, self.I)
                    user_degree = tf.squeeze(user_degree)
                    item_degree = tf.squeeze(item_degree)

            with tf.Session(config=config) as sess:
                self.user_degree, self.item_degree = sess.run([user_degree, item_degree],
                                                              feed_dict={self.edge_vals: self.edge_vals_d,
                                                                         self.edge_idx: self.edge_idx_d})

        
        print repr(np.sum(self.user_degree))
        print repr(np.sum(self.item_degree))

        self.occupied_pairs = edge_idx.shape[0] # oc_pa

        self._initialize_parameters(hparams, ppm)

        # random sample for diagnostics
        np.random.seed(self.seed)
        self.included_sample = self.edge_idx_d[np.random.choice(self.edge_idx_d.shape[0], 1000, replace=False)]
        user_sample = np.random.choice(self.U, 1000)
        item_sample = np.random.choice(self.I, 1000)
        self.pair_sample = np.vstack((user_sample, item_sample)).T

        # appx llhd for assessing convergence
        with tf.name_scope("appx_llhd"):
             self._build_appx_elbo()

        # computational graph for coordinate ascent
        with tf.name_scope("coordinate_ascent"):
            self._build_computation_graph()

        with tf.name_scope("evaluation"):
            with tf.device(self.device):
                self._build_predict_edges()
                self.edge_mean_summary = tf.reduce_mean(self.q_e_aux_vals.mean(), axis=0)

        with tf.name_scope("recommendation"), tf.device(self.device):
            self._build_rec_uncensored_edge_pops()

            self._censored_edge_pops = tf.placeholder(dtype=tf.float32)
            self._num_rec = tf.placeholder(dtype=tf.int32, shape=())
            self._top_k = tf.nn.top_k(self._censored_edge_pops, self._num_rec)

        # logging
        self.summary_writer = tf.summary.FileWriter('../logs', graph=self.sess.graph)

        # Initializing the tensor flow variables
        with tf.device(self.device):
            init = tf.global_variables_initializer()
        self.sess.run(init)

        # qm_du, qm_di were initialized arbitrarily, and are thus inconsistent w initialize value of the edge params
        # this line fixes that
        if not(ppm):
            self.sess.run(self.deg_update, feed_dict={self.edge_vals: self.edge_vals_d, self.edge_idx: self.edge_idx_d})


    def _initialize_parameters(self, hparams, ppm):

        K = np.float32(self.K)

        su, tu, a, b, self.size_u = (hparams['su'], hparams['tu'], hparams['a'], hparams['b'], hparams['size_u'])
        si, ti, c, d, self.size_i = (hparams['si'], hparams['ti'], hparams['c'], hparams['d'], hparams['size_i'])

        with tf.name_scope("hparams"), tf.device(self.device):
            ## Hyperparameters
            self.lsu = tf.Variable(softplus_inverse(-hparams['su'] + 1.), dtype=tf.float32, name="lsu")
            self.su = -tf.nn.softplus(self.lsu) + 1.

            self.tu = tf.Variable(hparams['tu'], dtype=tf.float32, name="tu")

            self.a = tf.Variable(hparams['a'], dtype=tf.float32, name="a")
            self.b = tf.Variable(hparams['b'], dtype=tf.float32, name="b")

            self.lsi = tf.Variable(softplus_inverse(-hparams['si'] + 1.), dtype=tf.float32, name="lsi")
            self.si = -tf.nn.softplus(self.lsi) + 1.

            self.ti = tf.Variable(hparams['ti'], dtype=tf.float32, name="ti")

            self.c = tf.Variable(hparams['c'], dtype=tf.float32, name="c")
            self.d = tf.Variable(hparams['d'], dtype=tf.float32, name="d")

        e = np.sum(self.edge_vals_d, dtype=np.float32)

        # initial values for total user and total item masses of type K
        # set st \sum_k tim_k * tum_k = e (which is in fact a bit higher than it oughta be)
        # and using item_mass / user_mass ~ item_size / user_size (which is only kind of true)
        tum_init = np.sqrt(self.size_u / self.size_i * e / K)
        tim_init = np.sqrt(self.size_i / self.size_u * e / K)

        with tf.name_scope("user_params"), tf.device(self.device):
            # shape params are read off immediately from update equations
            # rate params set to be consistent w \gam_i ~ 1, \sum_j beta_jk beta_k ~ \sqrt(e/k) (which is self consistent)
            if ppm :
                # If creating the principled predictive (ppm), don't have the user_degree. Just create some random initialization for now, we'll update it with a default value
                self.gam_shp = tf.Variable(tf.random_gamma([self.U, 1], 5., 5., seed=self.seed), dtype=tf.float32, name="gam_rte") 
                self.gam_rte = tf.Variable(tf.random_gamma([self.U, 1], 5., 5., seed=self.seed), dtype=tf.float32, name="gam_rte") 
                self.theta_shp = tf.Variable(tf.random_gamma([self.U, self.K], 10., 10., seed=self.seed), name="theta_shp")
                self.theta_rte =tf.Variable(tf.random_gamma([self.U, self.K], 5., 5., seed=self.seed), name="theta_rte") 
                self.g = tf.Variable(tf.random_gamma([self.K, 1], 0.001, 1, seed=self.seed) + TINY, name="g") 
            else:
                user_degs = np.expand_dims(self.user_degree, axis=1)
                self.gam_shp = tf.Variable((user_degs - su), name="gam_shp")  # s^U
                self.gam_rte = tf.Variable(np.sqrt(e) * (0.9 + 0.1*tf.random_gamma([self.U, 1], 5., 5., seed=self.seed)), dtype=tf.float32, name="gam_rte")  # r^U
                init_gam_mean = self.gam_shp.initial_value / self.gam_rte.initial_value
                self.theta_shp = tf.Variable((a + user_degs/K) * tf.random_gamma([self.U, self.K], 10., 10., seed=self.seed), name="theta_shp")  # kap^U
                self.theta_rte = tf.Variable((b + init_gam_mean * tim_init)*(0.9 + 0.1*tf.random_gamma([self.U, self.K], 5., 5., seed=self.seed)), name="theta_rte")  # lam^U
                self.g = tf.Variable(tf.random_gamma([self.K, 1], 0.001, 1, seed=self.seed) + TINY, name="g")  # g


        with tf.name_scope("item_params"), tf.device(self.device):
            ## Items
            if ppm:
                self.omega_shp = tf.Variable(tf.random_gamma([self.I, 1], 5., 5., seed=self.seed), name="omega_shp")  # s^I
                self.omega_rte = tf.Variable(tf.random_gamma([self.I, 1], 5., 5., seed=self.seed), dtype=tf.float32, name="omega_rte")  # r^I
                self.beta_shp = tf.Variable(tf.random_gamma([self.I, self.K], 10., 10., seed=self.seed), name="beta_shp")  # kap^I
                self.beta_rte = tf.Variable(tf.random_gamma([self.I, self.K], 5., 5., seed=self.seed), name="beta_rte")  # lam^I
                self.w = tf.Variable(tf.random_gamma([self.K, 1], 0.001, 1, seed=self.seed) + TINY, name="w")  # w
            else:
                item_degs = np.expand_dims(self.item_degree, axis=1)
                self.omega_shp = tf.Variable((item_degs - si), name="omega_shp")  # s^I
                self.omega_rte = tf.Variable(np.sqrt(e) * (0.9 + 0.1*tf.random_gamma([self.I, 1], 5., 5., seed=self.seed)), dtype=tf.float32, name="omega_rte")  # r^I
                init_omega_mean = self.omega_shp.initial_value / self.omega_rte.initial_value
                self.beta_shp = tf.Variable((c + item_degs/K) * tf.random_gamma([self.I, self.K], 10., 10., seed=self.seed), name="beta_shp")  # kap^I
                self.beta_rte = tf.Variable((d + init_omega_mean*tum_init) * (0.9 + 0.1*tf.random_gamma([self.I, self.K], 5., 5., seed=self.seed)), name="beta_rte")  # lam^I
                self.w = tf.Variable(tf.random_gamma([self.K, 1], 0.001, 1, seed=self.seed) + TINY, name="w")  # w

        with tf.device('/cpu:0'):
            with tf.variable_scope("edge_params", reuse=None):
                ## Edges
                if self.simple_graph:
                    # set init value so there's approximately 1 expected edge between each pair... WARNING: this may be profoundly stupid
                    self.sg_edge_param = tf.get_variable(name="sg_edge_param", shape=[self.occupied_pairs, self.K], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(mean=-np.log(K), stddev=1. / K, seed=self.seed),
                                    partitioner=tf.fixed_size_partitioner(self.edge_param_splits, 0))
                else:
                    self.lphi = tf.get_variable(name="lphi", shape=[self.occupied_pairs, self.K], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(mean=0, stddev=1. / K, seed=self.seed),
                                    partitioner=tf.fixed_size_partitioner(self.edge_param_splits, 0))

        with tf.name_scope("variational_post"), tf.device(self.device):

            # Variational posterior distributions
            self.q_gam = Gamma(concentration=self.gam_shp, rate=self.gam_rte, name="q_gam")
            self.q_theta = Gamma(concentration=self.theta_shp, rate=self.theta_rte, name="q_theta")
            self.q_g = PointMass(self.g, name="q_g")

            self.q_omega = Gamma(concentration=self.omega_shp, rate=self.omega_rte, name="q_omega")
            self.q_beta = Gamma(concentration=self.beta_shp, rate=self.beta_rte, name="q_beta")
            self.q_w = PointMass(self.w, name="q_w")

            if self.simple_graph:
                self.q_e_aux_vals = tPoissonMulti(log_lams=self.sg_edge_param, name="q_e_aux_vals") # q_edges_aux_flat
            else:
                self.q_e_aux_vals = Multinomial(total_count=self.edge_vals, logits=self.lphi, name="q_e_aux_vals") # q_edges_aux_flat
                self.q_e_aux_vals_mean = self.q_e_aux_vals.mean()

        with tf.name_scope("degree_vars"):
            # create some structures to make it easy to work with the expected value (wrt q) of the edges

            # qm_du[u,k] is the expected weighted degree of user u counting only edges of type k
            # qm_du[u,k] = E_q[e^k_i.] in the language of the paper
            # initialized arbitrarily, will override at end of init to set to
            # we use a tf.Variable here to cache the q_e_aux_vals.mean() value
            self.qm_du = tf.Variable(tf.ones([self.U, self.K], dtype=tf.float32), name="qm_du")
            self.qm_di = tf.Variable(tf.ones([self.I, self.K], dtype=tf.float32), name="qm_di")

        # Total Item Mass:
        self.i_tot_mass_m = self.q_w.mean() + tf.matmul(self.q_beta.mean(), self.q_omega.mean(), transpose_a=True)
        # Total User Mass:
        self.u_tot_mass_m = self.q_g.mean() + tf.matmul(self.q_theta.mean(), self.q_gam.mean(), transpose_a=True)

    def _build_computation_graph(self):
        with tf.name_scope("user_update"):
            with tf.device(self.device):
                [new_gam_shp, new_gam_rte,
                 new_theta_shp, new_theta_rte, new_g] = user_updates(
                    q_gam=self.q_gam, q_theta=self.q_theta, q_omega=self.q_omega, q_beta=self.q_beta, q_w=self.q_w,
                    qm_du=self.qm_du,
                    a=self.a, b=self.b, su=self.su, tu=self.tu,
                    size=self.size_u,
                    comp_rem=self.comp_rem,
                    n_samp=95
                    )

            # observation: gamma_rte depends on theta, and theta_rte depends on gamma
            # so these shouldn't update simultaneously
            # logical division: compute gamma update, then compute theta update.
            # we compute theta_shp as part of gamma update to avoid (huge) repeated computation
            self.u_update_one = assign_list(vars=[self.gam_shp, self.gam_rte, self.theta_shp, self.g],
                                            new_values=[new_gam_shp, new_gam_rte, new_theta_shp, new_g])
            self.u_update_two = assign_list(vars=[self.theta_rte], new_values=[new_theta_rte])

        with tf.name_scope("item_update"):
            with tf.device(self.device):
                [new_omega_shp, new_omega_rte,
                 new_beta_shp, new_beta_rte, new_w] = user_updates(
                    self.q_omega, self.q_beta,
                    self.q_gam, self.q_theta, self.q_g,
                    self.qm_di,
                    self.c, self.d, self.si, self.ti,
                    size=self.size_i,
                    comp_rem=self.comp_rem,
                    n_samp=95)

            # division into two updates for same reason as users
            self.i_update_one = assign_list(vars=[self.omega_shp, self.omega_rte, self.beta_shp, self.w],
                             new_values=[new_omega_shp, new_omega_rte, new_beta_shp, new_w])
            self.i_update_two = assign_list(vars=[self.beta_rte], new_values=[new_beta_rte])

        with tf.name_scope("edge_update"):
            with tf.device(self.device):
                # split the edge list to avoid memory issues
                edge_idx_split = tensor_split(self.edge_idx, self.edge_param_splits)

                if self.simple_graph:
                    new_sg_edge_params = \
                        [simple_graph_edge_update(self.q_theta, self.q_beta, self.q_gam, self.q_omega, edge_idx) for edge_idx in edge_idx_split]
                else:
                    new_lphis = \
                        [multi_graph_edge_update(self.q_theta, self.q_beta, edge_idx) for edge_idx in edge_idx_split]

            if self.simple_graph:
                self.sg_edge_param_update = [sg_edge_param.assign(new_sg_edge_param) for sg_edge_param, new_sg_edge_param
                                             in zip(self.sg_edge_param._get_variable_list(), new_sg_edge_params)]
            else:
                self.lphi_update = [lphi.assign(new_lphi) for (lphi, new_lphi) in zip(self.lphi._get_variable_list(), new_lphis)]

        with tf.name_scope("qm_deg_update"):
            with tf.device(self.device):
                new_qm_du, new_qm_di = compute_degrees2(self.q_e_aux_vals.mean(), self.edge_idx, self.U, self.I)
            
            self.deg_update = assign_list(vars=[self.qm_du, self.qm_di], new_values=[new_qm_du, new_qm_di])


    def _fix_post_assigns(self, true_omega, true_beta, users=False, items=True):
        """
        Method to be used for debugging:
        Fix item parameters to ground truth values
        """
        # fix variational posteriors to be tightly concentrated around true values

        item_assigns = assign_list([self.omega_shp, self.omega_rte, self.beta_shp, self.beta_rte, self.w],
                         [100 * true_omega, 100 * tf.ones_like(true_omega),
                       100 * true_beta, 100 * tf.ones_like(true_beta),
                       0.01 * tf.ones_like(self.w)]) # actually, I'm not sure about this one
        self.sess.run(item_assigns, feed_dict={self.edge_vals: self.edge_vals_d, self.edge_idx: self.edge_idx_d})

    def _edge_prob_samples(self, pred_edges, N=100):
        """

        :param pred_edges: edge list
        :param N: number of samples
        :param log: if True, return E[log(p(e_ij = 1 | params)]
        :return: E[p(e_ij = 1 | sampled_params)] for each ij in pred_edges
        """
        users_idx = pred_edges[:, 0]
        items_idx = pred_edges[:, 1]

        # MC estimate
        # this is logically equivalent to drawing samples from q and then gathering the necessary ones,
        # but much faster when there are many users and items

        # relevant params for simulation
        omega_shp = tf.gather(self.omega_shp, items_idx)
        omega_rte = tf.gather(self.omega_rte, items_idx)
        beta_shp = tf.gather(self.beta_shp, items_idx)
        beta_rte = tf.gather(self.beta_rte, items_idx)

        gam_shp = tf.gather(self.gam_shp, users_idx)
        gam_rte = tf.gather(self.gam_rte, users_idx)
        theta_shp = tf.gather(self.theta_shp, users_idx)
        theta_rte = tf.gather(self.theta_rte, users_idx)

        # samples for MC estimate
        omega_smp = tf.random_gamma([N], omega_shp, omega_rte, seed=self.seed)
        beta_smp = tf.random_gamma([N], beta_shp, beta_rte, seed=self.seed)
        gam_smp = tf.random_gamma([N], gam_shp, gam_rte, seed=self.seed)
        theta_smp = tf.random_gamma([N], theta_shp, theta_rte, seed=self.seed)

        user_weights_s = gam_smp * theta_smp
        item_weights_s = omega_smp * beta_smp
        edge_weight_s = tf.reduce_sum(user_weights_s * item_weights_s, axis=2)

        prob_samp = 1. - tf.exp(-edge_weight_s)
        return prob_samp

    def _build_predict_edges(self, N=100):
        """
        Only handles SG
        Returns prob given an edge list
        """
        with tf.device(self.device):
            self.pred_edges_ph = tf.placeholder(dtype=tf.int32)
            # MC estimate
            self.predict_edges = tf.reduce_mean(self._edge_prob_samples(self.pred_edges_ph, N=N), axis=0)


    def _build_rec_uncensored_edge_pops(self):
        """
        Builds matrix of expected number of edges between all items and self._rec_users
        """
        with tf.device(self.device):
            self._rec_users = tf.placeholder(dtype=tf.int32)

            q_gam_mean = self.q_gam.mean()
            q_theta_mean = self.q_theta.mean()

            q_omega_mean = self.q_omega.mean()
            q_beta_mean = self.q_beta.mean()

            user_params = tf.gather(q_gam_mean, self._rec_users) * tf.gather(q_theta_mean, self._rec_users)
            item_params  = q_omega_mean * q_beta_mean

            # edge_pops[user,item] gives the affinity of user to item
            self._rec_uncensored_edge_pops = tf.matmul(user_params, item_params, transpose_b=True)

    def _build_appx_elbo(self):
        """
        Returns an estimate of \sum_{e in test_idxs} log(prob(e)) + \sum{e not in test_idxs} log(1-prob(e))
        this is not actually the log likelihood because it ignores the contribution of uninstantiated atoms
        (actually, maybe this is handled after all...)
        :param test_idxs: tensor of shape [e, 2], indices of edges of graph
        :return: estimate of \sum_{e in test_idxs} log(prob(e)) + \sum{e not in test_idxs} log(1-prob(e))
        """

        # MC estimate of contribution from edges
        # obvious choice: uniformly sample terms... but resulting estimator is super high variance
        # edges_sample = np.copy(self.edge_idx_d[np.random.choice(self.edge_idx_d.shape[0], 3000, replace=False)]).astype(np.int32)
        # so instead use p-sampling... although it's unclear whether this really represents a major improvement
        e = self.edge_vals_d.shape[0]
        p_inc = np.sqrt(5000. / e) #use about 5000 edges for MC est
        edges_sample = item_p_sample(user_p_sample(self.edge_idx_d, p_inc)[0], p_inc)[0].astype(np.int32)

        # clip by value because of numerical issues
        p_edge_samples = tf.clip_by_value(self._edge_prob_samples(edges_sample), 1e-15, 1.)

        # reduce_mean is MC estimate over params of model, reduce_sum is summing cont from p-samp
        edge_llhd_est = 1. / p_inc**2 * tf.reduce_sum(tf.reduce_mean(tf.log(p_edge_samples), axis=0))

        # log(1-p_ij) = -lambda_ij, so:
        tot_lam_sum = tf.reduce_sum(self.i_tot_mass_m*self.u_tot_mass_m) # includes contribution from edges as well as non-edges
        # subtract off edge contribution:
        user_params = tf.gather(self.q_gam.mean() * self.q_theta.mean(), self.edge_idx[:,0])
        item_params = tf.gather(self.q_omega.mean() * self.q_beta.mean(), self.edge_idx[:,1])
        edges_lam_sum = tf.reduce_sum(user_params * item_params)
        nonedge_llhd_term = -(tot_lam_sum - edges_lam_sum)

        # hopefully lower variance than direct MC est
        #\sum_edges log(p_ij) = -\sum_edges lam_ij + \sum_ij log(p_ij / (1-p_ij))

        # note: the reduce mean here averages over both the sampled params in p_edge_samples, and over the random choice of edges
        # edge_llhd_est = -edges_lam_sum + e*tf.reduce_mean(tf.reduce_mean(tf.log(p_edge_samples / (1. - p_edge_samples)), axis=0))

        self.appx_elbo = [edge_llhd_est, nonedge_llhd_term]



    def load_pretrained_model(self, gam_shp, gam_rte, theta_shp, theta_rte, g, omega_shp, omega_rte, beta_shp, beta_rte, w):
        user_assign = assign_list([self.gam_shp, self.gam_rte, self.theta_shp, self.theta_rte, self.g],
                                           [gam_shp, gam_rte, theta_shp, theta_rte, g])
        item_assign = assign_list([self.omega_shp, self.omega_rte, self.beta_shp, self.beta_rte, self.w],
                                           [omega_shp, omega_rte, beta_shp, beta_rte, w])

        if self.simple_graph:
            self.sess.run(self.sg_edge_param_update, feed_dict={self.edge_idx: self.edge_idx_d})
        else:
            self.sess.run(self.lphi_update, feed_dict={self.edge_idx: self.edge_idx_d})

        self.sess.run(self.deg_update, feed_dict={self.edge_vals: self.edge_vals_d, self.edge_idx: self.edge_idx_d})
        pass

    def infer(self, n_iter=150):
        """
        Runs the co-ordinate ascent inference on the model. 
        """
        if self.ppm:
            print("Running infer is forbidden for principled predictive model.")
            return
        if DEBUG:
            # fix some variables to their true values
            self._fix_post_assigns(self.ground_truth['true_omega'], self.ground_truth['true_beta'])

        with self.sess.as_default():
            for i in range(n_iter):

                # users
                start_time = time.time()
                self.sess.run(self.u_update_one, feed_dict={self.edge_idx: self.edge_idx_d})
                self.sess.run(self.u_update_two, feed_dict={self.edge_idx: self.edge_idx_d})

                # items
                if not(self.fix_item_params):
                    start_time = time.time()
                    self.sess.run(self.i_update_one, feed_dict={self.edge_idx: self.edge_idx_d})
                    self.sess.run(self.i_update_two, feed_dict={self.edge_idx: self.edge_idx_d})

                # edges
                start_time = time.time()
                if self.simple_graph:
                    for sg_edge_param_update in self.sg_edge_param_update:
                        self.sess.run(sg_edge_param_update, feed_dict={self.edge_idx: self.edge_idx_d})
                else:
                    for lphi_update in self.lphi_update:
                        self.sess.run(lphi_update, feed_dict={self.edge_idx: self.edge_idx_d})

                # mean degree (caching)
                start_time = time.time()
                self.sess.run(self.deg_update, feed_dict={self.edge_vals: self.edge_vals_d, self.edge_idx: self.edge_idx_d})

                ### Print the total item and user mass ###
                if np.mod(i, 30) == 0:
                    self._logging(i)
                    print("appx_elbo: {}".format(self.sess.run(self.appx_elbo,
                                                           feed_dict={self.edge_idx: self.edge_idx_d})))

            ## DONE TRAINING
            self.user_affil_est = to_prob(self.theta_shp / self.theta_rte).eval()
            self.item_affil_est = to_prob(self.beta_shp / self.beta_rte).eval()
            if DEBUG: 
                self.true_user_affil = to_prob(self.ground_truth['true_theta']).eval()
                self.true_item_affil = to_prob(self.ground_truth['true_beta']).eval()

            # User params
            gam_shp, gam_rte, theta_shp, theta_rte, g = self.sess.run([self.gam_shp, self.gam_rte, self.theta_shp, self.theta_rte, self.g])

            # Item params
            omega_shp, omega_rte, beta_shp, beta_rte, w = self.sess.run([self.omega_shp, self.omega_rte, self.beta_shp, self.beta_rte, self.w])

            return gam_shp, gam_rte, theta_shp, theta_rte, g, omega_shp, omega_rte, beta_shp, beta_rte, w


    def test_llhd(self, test_idxs):
        """
        Returns an estimate of \sum_{e in test_idxs} log(prob(e)) and of \sum{e not in test_idxs} log(1-prob(e))
        :param test_idxs: tensor of shape [e, 2], indices of edges of graph
        :return: estimate of [\sum_{e in test_idxs} log(prob(e)), \sum{e not in test_idxs} log(1-prob(e))]
        """

        test_idxs_ = np.copy(test_idxs)

        users = np.unique(test_idxs_[:, 0])
        train_idxs = np.copy(self.edge_idx_d[np.in1d(self.edge_idx_d[:, 0], users),:])

        for en, user in enumerate(users):
            test_idxs_[test_idxs_[:, 0] == user, 0] = en
            train_idxs[train_idxs[:,0] == user, 0] = en

        matrix = np.ones((users.shape[0], self.I))
        matrix[train_idxs.T.tolist()] = 0
        matrix[test_idxs_.T.tolist()] = 0
        all_but_test_idxs = np.array(matrix.nonzero()).T

        # Select 1000 edges randomly from test_idx to get an estimate of the expected value
        np.random.seed(self.seed)
        selected_edges = np.random.choice(test_idxs_.shape[0], min(1000, test_idxs_.shape[0]), replace=False)
        test_idxs_ = test_idxs_[selected_edges]

        # Select 1000 edges randomly from all_but_traintest_idx to get an estimate of the expected value
        selected_edges = np.random.choice(all_but_test_idxs.shape[0], min(1000, all_but_test_idxs.shape[0]), replace=False)
        all_but_test_idxs = all_but_test_idxs[selected_edges]

        p_test_idx = self.sess.run(self.predict_edges, feed_dict={self.pred_edges_ph: test_idxs_})
        p_not_test_idx = self.sess.run(self.predict_edges, feed_dict={self.pred_edges_ph: all_but_test_idxs})

        return np.mean(np.log(p_test_idx)), np.mean(np.log(1.-p_not_test_idx))


    def recommend(self, K, users=None, excluded_items=None):
        """
        Recommend Top-K Items
        NOTE: Does not censor train edges while recommending
        Reasoning: If we pass holdout set as train data and do not run infer, 
                   then we don't want to censor "train" edges while making recommendations
        outputs top K recommendations for each user in users

        Warning: assumes number of items > K

        :param users: numpy array, users to make recommendations for
        :param K: number of recommendations to output
        :param excluded_items: (optional) numpy array, items to exclude from recommendations

        """
        # sort users and remove redundant (for easy 0-indexing later)
        # uniq_inv will be used to restore original ordering at output
        if users is None:
            users_ =  np.unique(self.edge_idx_d[:,0])
            uniq_inv = range(users_.shape[0])
        else:
            users_, uniq_inv = np.unique(users, return_inverse=True)

        # probability of connection for each user in users and all items
        edge_pops = self.sess.run(self._rec_uncensored_edge_pops, feed_dict={self._rec_users: users_})
        # do any necessary additional censoring
        if excluded_items is not None:
            edge_pops[:, excluded_items] = 0.

        recs = self.sess.run(self._top_k, feed_dict={self._censored_edge_pops: edge_pops, self._num_rec: K})
        # restore original ordering
        recs_orig_ordering = recs
        recs_orig_ordering._replace(indices=recs.indices[uniq_inv, :])
        return recs_orig_ordering


    def nDCG(self, p, users=None, test=None, ranks=None, excluded_items=None):
        """
        Computes the normalized Discounted Cumulative Gain at rank p
        """

        # returns a sorted array
        if users is None:
            users = np.unique(self.edge_idx_d[:,0])
        if test is None:
            test = self.edge_idx_d
        if ranks is None:
            ranks = self.recommend(p, users, excluded_items).indices

        nDCG = np.zeros(users.shape[0])
        for en, user in enumerate(users):
            user_test = np.copy(test[test[:, 0] == user])
            test_ranks = np.isin(ranks[en, :], user_test[:, 1]).nonzero()[0] + 1
            DCG = np.sum(np.log(2.) / (np.log(test_ranks + 1)))
            num_rel = min(p, user_test.shape[0])  # number of relevant
            itest_ranks = np.array(range(num_rel)) + 1
            iDCG = np.sum(np.log(2.) / (np.log(itest_ranks + 1)))
            nDCG[en] = DCG / iDCG
        return nDCG

    def sample_one(self, user_size = None, item_size = None, eps=1e-8):
        """
        Draw a posterior sample from the fitted model

        :param user_size: float, user size
        :param item_size: float, item size
        :param eps: float, approximation level for ggp (default 1e-8); atom weights smaller than this are ignored
        :return: An approximate sample of the multigraph with the associated parameters
        a numpy array [user_idx, item_idx, num_edges] of length equal to the total occupied pairs
        """

        if user_size is None:
            _user_size = self.size_u
        else:
            _user_size = user_size

        if item_size is None:
            _item_size = self.size_i
        else:
            _item_size = item_size

        i_mass_samp = self.sess.run(self.q_omega.sample(seed=self.seed) * self.q_beta.sample(seed=self.seed))
        u_mass_samp = self.sess.run(self.q_gam.sample(seed=self.seed) * self.q_theta.sample(seed=self.seed))

        i_mass_tots = np.sum(i_mass_samp, 0) # total mass of each type in items
        u_mass_tots = np.sum(u_mass_samp, 0)

        """
        edges between instantiated vertices
        """
        # total number of edges of each type
        tot_edges_mean = u_mass_tots * i_mass_tots
        tot_edges = np.random.poisson(tot_edges_mean)

        # K probability distributions over items / users
        i_probs = i_mass_samp / i_mass_tots
        i_probs[i_probs < 1e-8] = 0 # numerical precision hack
        u_probs = u_mass_samp / u_mass_tots
        u_probs[u_probs < 1e-8] = 0 # numerical precision hack

        # assign edges to pairs
        item_assignments = [np.random.choice(self.I, size=tot_edges[k],replace=True,p=i_probs[:,k]) for k in range(self.K)]
        user_assignments = [np.random.choice(self.U, size=tot_edges[k],replace=True,p=u_probs[:,k]) for k in range(self.K)]

        edge_list = np.concatenate(
            [np.vstack([user_assignments[k], item_assignments[k]]) for k in range(self.K)], -1).T

        """
        leftover mass contribution
        Approximation: uninstantiated points never connect to each other
        """
        if _item_size != 0:

            # total mass belonging to uninstantiated items in each dimension
            rem_item_mass = (_item_size / self.size_i) * self.sess.run(
                self.q_w.sample(seed=self.seed)[:, 0])  # since q_w = size * rate

            # number of edges between instantiated users and uninstantiated items
            n_insu_remi = np.random.poisson(u_mass_tots * rem_item_mass)

            # ids of users connecting to uninstantiated atoms
            u_assign = np.concatenate([np.random.choice(self.U, size=n_insu_remi[k], replace=True, p=u_probs[:, k]) for k in range(self.K)])

            """
            it remains to assign the termini to atoms in the uninstantiated part of the marked GGPs
            strategy: simulate the posterior marked GGPs, and use the same multinomial assignment
            warning: this is computationally pricey
            """

            # sample from the point process of atoms that failed to connect to anything when the dataset was originally generated
            si, ti, c, d = self.sess.run([self.si, self.ti, self.c, self.d])
            new_ggp = sample_ggp(_item_size, si, ti, eps)
            sim_marks = np.random.gamma(shape=c, scale=1./d, size=new_ggp.shape + (self.K,))
            atom_weights = np.expand_dims(new_ggp,1) * sim_marks

            # uninstantiated atoms
            not_inc_prob = np.exp(-np.sum(atom_weights * u_mass_tots, axis=1)) # probability each item atom failed to connect to any user
            uninstant_atom_weights = atom_weights[np.nonzero(np.random.binomial(1,p=not_inc_prob))] # weights
            uninstant_atom_dist = uninstant_atom_weights / np.sum(uninstant_atom_weights, 0) # K probability dists

            # assign edges to these new atoms in the usual multinomial way
            i_rem_assign = np.concatenate([np.random.choice(uninstant_atom_dist.shape[0], size=n_insu_remi[k], replace=True, p=uninstant_atom_dist[:, k]) for k in range(self.K)])
            # these atoms should have labels not already taken by any previously instantiated atom
            i_rem_assign += self.I

            # and now compile the edge list
            insu_remi = np.vstack([u_assign , i_rem_assign]).T
            edge_list = np.concatenate([edge_list, insu_remi], axis=0)

        # repeat this for instantiated items + remaining users
        if _user_size != 0:

            rem_user_mass = (_user_size / self.size_u) * self.sess.run(self.q_g.sample(seed=self.seed)[:, 0])

            # number of edges connecting to previously uninstantiated atoms
            n_insi_remu = np.random.poisson(i_mass_tots * rem_user_mass)  # instantiated items, remaining users

            # ids of atoms connecting to uninstantiated users
            i_assign = np.concatenate([np.random.choice(self.I, size=n_insi_remu[k], replace=True, p=i_probs[:, k]) for k in range(self.K)])

            """
            assign the termini to atoms in the uninstantiated part of the marked GGPs
            """

            # sample from the point process of atoms that failed to connect to anything when the dataset was originally generated
            su, tu, a, b = self.sess.run([self.su, self.tu, self.a, self.b])
            new_ggp = sample_ggp(_user_size, su, tu, eps)
            sim_marks = np.random.gamma(shape=a, scale=1./b, size=new_ggp.shape + (self.K,))
            atom_weights = np.expand_dims(new_ggp,1) * sim_marks
            not_inc_prob = np.exp(-np.sum(atom_weights * i_mass_tots, axis=1)) # probability each user atom failed to connect to any item
            # uninstantiated atoms
            uninstant_atom_weights = atom_weights[np.nonzero(np.random.binomial(1,p=not_inc_prob))] # weights
            uninstant_atom_dist = uninstant_atom_weights / np.sum(uninstant_atom_weights, 0) # K probability dists

            # now assign to these new atoms in the usual multinomial way
            u_rem_assign = np.concatenate([np.random.choice(uninstant_atom_dist.shape[0], size=n_insi_remu[k], replace=True, p=uninstant_atom_dist[:, k]) for k in range(self.K)])
            # these atoms should have labels not already taken by any previously instantiated atom
            u_rem_assign += self.U

            # and now do the edge assignment
            insi_remu = np.vstack([u_rem_assign, i_assign]).T
            edge_list = np.concatenate([edge_list, insi_remu], axis=0)

        # cleanup
        uniques = np.unique(edge_list, return_counts=True, axis=0)
        return np.hstack([uniques[0], np.expand_dims(uniques[1], 1)])


    def principled_predictive_model(self, test_look_edge_idx, test_look_edge_vals, test_holdout,  user_update_iters=100, p=0.8,
                                free_model_resources = True, device='/cpu:0', seed=None):
        """
        Idea: data is originally divided into a test and train set using p-sampling of the users. Test set is further
         divided into test_lookup and test_holdout using p-sampling of the items in test set. 
         The object owning this method has been trained on the train set.

         We now further divide the test set into test_look, which will be used to propagate the fitted model to get
         parameter values for the users in the test set, and test_holdout, which we use to assess our algorithm using 
         item .

         This function returns a GNMF object on [items in test, users in test] to be used for further prediction.
         The item parameters are inherited from the trained model. The user parameters are set to be compatible with
         the item parameters using the usual edge+user update scheme.

        Remark: we return a model that includes all the items (rather than just the ones in test_look) because the items
        in test_holdout generally contain items not in test_look

        :param test_look_edge_idx:
        :param test_look_edge_vals:
        :param user_update_iters: number of iterations used to set users to be compatible w items
        :return:
        """
        """
        WARNING: self.hparams doesn't reflect any updates that have happened to the hyperparams, so if we ever write
        tuning code we'll have to be cognizant of this
        """
        U = np.unique(test_look_edge_idx[:, 0]).shape[0]
        I = np.unique(test_look_edge_idx[:, 1]).shape[0]
        """
        First, propogate the trained item values to the test set users
        """
        lookup_items = np.unique(test_look_edge_idx[:,1]) # items connected to any lookup user
        lookup_I = lookup_items.shape[0]

        # make the lookup item labels contiguous for passing into GNMF (zero indexing)      
        [lookup_relabel, convert] = zero_index(test_look_edge_idx, 1)

        with tf.variable_scope("holdout_fitter"):
            holdout_hparams = self.hparams.copy()
            holdout_hparams['size_i'] = p * self.hparams['size_i']
            holdout_hparams['size_u'] = (1.-p) / p * self.hparams['size_u']

            with GraphexNMF(lookup_relabel, test_look_edge_vals, U, lookup_I,  self.K, holdout_hparams,
                                        ground_truth=None, simple_graph=self.simple_graph, GPU=self.GPU,
                                        comp_rem=False, # comp_rem won't work because item weights are wrong
                                        fix_item_params=True, device=device, seed=seed) \
                    as holdout_fitter:

                # item parameters for the items in the lookup set
                omega_shp_lookup_op = tf.gather(self.omega_shp, lookup_items)
                omega_rte_lookup_op = tf.gather(self.omega_rte, lookup_items)
                beta_shp_lookup_op = tf.gather(self.beta_shp, lookup_items)
                beta_rte_lookup_op = tf.gather(self.beta_rte, lookup_items)
                w_lookup_op = p * self.w # w is implicitly item size times w, size transforms as s -> p*s under p-sampling

                # run it
                [omega_shp_lookup, omega_rte_lookup, beta_shp_lookup, beta_rte_lookup, w_lookup] = self.sess.run(
                    [omega_shp_lookup_op, omega_rte_lookup_op, beta_shp_lookup_op, beta_rte_lookup_op, w_lookup_op])

                # fix the item parameters to the fitted values
                item_assign = assign_list([holdout_fitter.omega_shp, holdout_fitter.omega_rte, holdout_fitter.beta_shp, holdout_fitter.beta_rte, holdout_fitter.w],
                                           [omega_shp_lookup, omega_rte_lookup, beta_shp_lookup, beta_rte_lookup, w_lookup])
                holdout_fitter.sess.run(item_assign)

                # infer the user parameters
                holdout_fitter.infer(user_update_iters)

                [fit_gam_shp, fit_gam_rte, fit_theta_shp, fit_theta_rte, fit_g] = holdout_fitter.sess.run(
                    [holdout_fitter.gam_shp, holdout_fitter.gam_rte, holdout_fitter.theta_shp, holdout_fitter.theta_rte, holdout_fitter.g])
        
        """
        Next, return the model that we'll use for prediction by taking the item values from the original trained model,
        and the user values from the holdout_fitter
        """ 
        test_holdout_users = np.unique(test_holdout[:,0])
        test_holdout_items = np.unique(test_holdout[:,1])
        holdout_U = test_holdout_users.shape[0]
        holdout_I = test_holdout_items.shape[0]
        # fix the item parameters to the fitted values - only users in holdout
        [omega_shp, omega_rte, beta_shp, beta_rte, w] = \
            self.sess.run([self.omega_shp, self.omega_rte, self.beta_shp, self.beta_rte, self.w])
        omega_shp = omega_shp[test_holdout_items]
        omega_rte = omega_rte[test_holdout_items]
        beta_shp = beta_shp[test_holdout_items]
        beta_rte = beta_rte[test_holdout_items]
        # fix the user parameters to the fitted values - only users in holdout
        gam_shp = fit_gam_shp[test_holdout_users]
        gam_rte = fit_gam_rte[test_holdout_users]
        theta_shp = fit_theta_shp[test_holdout_users]
        theta_rte = fit_theta_rte[test_holdout_users]
        g = fit_g
        # make the holdout item labels contiguous for passing into GNMF (zero indexing)      
        [holdout_relabel, convert_users] = zero_index(test_holdout, 0)
        [holdout_relabel, convert_items] = zero_index(holdout_relabel, 1)
        # release the session to free up resources to do recommendation with.
        # this is a bit nasty, and is used in part 'cause sess.close() doesn't work properly
        # WARNING: I'm not sure what happens if this command is run on a server... might be a good way of making enemies
        if free_model_resources : tf.Session.reset(None)

        with tf.variable_scope("ppm_init"):
            ppm_hparams = self.hparams.copy()
            ppm_hparams['size_i'] = (1.-p) * ppm_hparams['size_i'] # 1-p of items get into the holdout
            ppm_hparams['size_u'] = (1.-p) / p * ppm_hparams['size_u'] # 1-p of users get into the holdout (and self.hparams[size_u] is size of *train*)

            # passing in holdout data so we can compute appx_llhd for holdout.
            # IMPORTANT: MUST NOT RUN ppm.infer()!!
            # We do not want to update user and item parameters based on holdout dataset
            # Holdout is strictly for testing
            ppm = GraphexNMF(holdout_relabel[:,:2], holdout_relabel[:,2], holdout_U, holdout_I, self.K, ppm_hparams,
                                        ground_truth=None, simple_graph=self.simple_graph, GPU=self.GPU,
                                        comp_rem=False, fix_item_params=False,
                                        device=device, seed=seed, ppm=True)

            # correct size factor on w
            w = (1. - p) * w

            ppm_item_assign = assign_list(
                [ppm.omega_shp, ppm.omega_rte, ppm.beta_shp, ppm.beta_rte, ppm.w],
                [omega_shp, omega_rte, beta_shp, beta_rte, w])

            # correct size factor on g
            fit_g = (1. - p) / p * fit_g

            # fix the user parameters to the fitted values
            ppm_user_assign = assign_list(
                [ppm.gam_shp, ppm.gam_rte, ppm.theta_shp, ppm.theta_rte, ppm.g],
                [gam_shp, gam_rte, theta_shp, theta_rte, g])

            ppm.sess.run([ppm_user_assign, ppm_item_assign])

            # edge updates... strictly speaking, this doesn't matter for recommendations
            if ppm.simple_graph:
                ppm.sess.run(ppm.sg_edge_param_update, feed_dict={ppm.edge_idx: ppm.edge_idx_d})
            else:
                ppm.sess.run(ppm.lphi_update, feed_dict={ppm.edge_idx: ppm.edge_idx_d})
            ppm.sess.run(ppm.deg_update, feed_dict={ppm.edge_vals: ppm.edge_vals_d, ppm.edge_idx: ppm.edge_idx_d})

        return ppm


    def _logging(self, itr):
        print("----------------------------------------------------------------")
        print("ITERATION #{}".format(itr))
        print("mean community edge weights:{}").format(
            self.sess.run(self.edge_mean_summary, feed_dict={self.edge_vals: self.edge_vals_d, self.edge_idx: self.edge_idx_d}))
        print("----------------------------------------------------------------")
        print("P(inclusion | included): {}").format(
            np.mean(self.sess.run(self.predict_edges, feed_dict={self.pred_edges_ph: self.included_sample})))
        print("----------------------------------------------------------------")
        print("P(inclusion | random pair): {}").format(
            np.mean(self.sess.run(self.predict_edges, feed_dict={self.pred_edges_ph: self.pair_sample})))
        print("----------------------------------------------------------------")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

