import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib

from GraphexNMF import GraphexNMF
from plotting.plotting_helpers import plot_deg_counts
from plotting.plotting_helpers import plot_deg_counts_users, plot_deg_counts_items
from hyperparam_ests import estimate_sigmas_det, estimate_size
from helpers import zero_index

"""
Runs the model on the specified dataset
Takes datadir as an argument
Takes hyperparameters for the model
"""

def main(train, lookup, holdout, hparams, gpu_flag, simple_graph, data_dir="./", comp_rem=True, seed=None):
    [tu, su, a, b,ti, si, c, d,K] = [hparams['tu'], hparams['su'], hparams['a'], hparams['b'],hparams['ti'], hparams['si'], hparams['c'], hparams['d'],hparams['K']]
    [size_u , size_i] = [hparams['size_u'] , hparams['size_i']]

    if gpu_flag:
        device = '/gpu:0'
        device2 = '/gpu:1' ## We have enabled soft placement in case gpu:1 doesn't exist
    else:
        device = device2 = '/cpu:0'
    sparse = hparams['sparse']


    U = np.unique(train[:,0]).shape[0]
    I = np.unique(train[:,1]).shape[0]
    lookup_U = np.unique(lookup[:,0]).shape[0]
    lookup_I = np.unique(lookup[:,1]).shape[0]
    holdout_U = np.unique(holdout[:,0]).shape[0]
    holdout_I = np.unique(holdout[:,1]).shape[0]

    # test metric
    ndcg_p = min(hparams['nDCG'], holdout_I)
    topk = min(hparams['topK'], holdout_I)

    ## INSTANTIATE + TRAIN MODEL
    model = GraphexNMF(train[:,:2].astype(int), train[:,2].astype(int), U, I, hparams['K'], hparams,
                       simple_graph=simple_graph, GPU=gpu_flag, comp_rem=comp_rem, seed=seed, device=device, edge_param_splits=1)

    [gam_shp_tr, gam_rte_tr, theta_shp_tr, theta_rte_tr, g_tr, omega_shp_tr, omega_rte_tr, beta_shp_tr, beta_rte_tr, w_tr] = model.infer(hparams['itr'])
    model._logging(" converged model")

    ## Save trained model
    with open(data_dir+"/trained_model.pkl", "wb") as f:
        pickle.dump(gam_shp_tr, f)
        pickle.dump(gam_rte_tr, f)
        pickle.dump(theta_shp_tr, f)
        pickle.dump(theta_rte_tr, f)
        pickle.dump(g_tr, f)
        pickle.dump(omega_shp_tr, f)
        pickle.dump(omega_rte_tr, f)
        pickle.dump(beta_shp_tr, f)
        pickle.dump(beta_rte_tr, f)
        pickle.dump(w_tr, f)

    ## PLOT + SAVE DEGREE DISTRIBUTION FOR TRAINED MODEL
    post_sample_train = model.sample_one()
    with open(data_dir+"/post-tr.pkl", "wb") as f:
        pickle.dump(post_sample_train, f)
    
    # Users
    plt.clf()
    plot_deg_counts_users(train)
    plot_deg_counts_users(post_sample_train, color='green')
    plt.savefig(data_dir+"/post_samp_train_users")
    # Items
    plt.clf()
    plot_deg_counts_items(train)
    plot_deg_counts_items(post_sample_train, color='green')
    plt.savefig(data_dir+"/post_samp_train_items")


    ### TESTING
    ## Dataset Processing
    # List of test items with where indexing corresponds to train set
    holdout_users = np.unique(holdout[:,0]).copy()
    holdout_items = np.unique(holdout[:,1]).copy()


    ## Train principled predictive model
    # IMPORTANT!: Passing holdout to create ppm. but we should not run infer, since that would be cheating.
    ppm = model.principled_predictive_model(lookup[:,0:2], lookup[:,2], holdout,
                                        free_model_resources=False, user_update_iters=int(hparams['itr']*0.75), seed=seed, device=device2)
    ppm._logging(" converged predictive model")

    ##
    # Updating holdout to zero indexed version
    # this must hold: edges_idx_d == holdout_relabel 
    [holdout_relabel, convert_users] = zero_index(holdout, 0)
    [holdout_relabel, convert_items] = zero_index(holdout_relabel, 1)

    post_sample_holdout = ppm.sample_one()
    with open(data_dir+"/post-ho.pkl", "wb") as f:
        pickle.dump(post_sample_holdout, f)

    # Users
    plt.clf()
    plot_deg_counts_users(holdout_relabel)
    plot_deg_counts_users(post_sample_holdout, color='green')
    plt.savefig(data_dir+"/post_samp_holdout_users")

    # Items
    plt.clf()
    plot_deg_counts_items(holdout_relabel)
    plot_deg_counts_items(post_sample_holdout, color='green')
    plt.savefig(data_dir+"/post_samp_train_items")

    ## Computing Different Recommendation Metrics
    #
    print("Computing Recommendation Metrics")
    holdout_users_relabel = np.unique(holdout_relabel[:,0])
    holdout_items_relabel = np.unique(holdout_relabel[:,1])
    holdout_I = holdout_items_relabel.shape[0]
    # Note: We need to create a relabeled version of holdout so the 


    # Precision + nDCG
    ranks = ppm.recommend(ndcg_p).indices
    recommendations = ranks[:,:topk]

    ndcg = ppm.nDCG(ndcg_p, ranks=ranks)
    precision = np.zeros(holdout_users_relabel.shape[0])
    for en,user in enumerate(holdout_users_relabel):
        user_test_edges = np.copy(holdout_relabel[holdout_relabel[:, 0] == user, 1])
        precision[en] = float(np.intersect1d(user_test_edges, recommendations[en, :]).shape[0]) / float(np.minimum(20., float(user_test_edges.shape[0])))

    # Unpopular Precision
    sid = np.sort(model.item_degree[model.item_degree > 0])
    thr = sid[int(hparams['unpop'] * sid.shape[0])]
    pop_items = (model.item_degree > thr).nonzero()[0]
    ## Keeping these 2  versions around for plotting. Probably should think of a better way.
    pop_items = np.intersect1d(pop_items,np.unique(holdout[:,1]))
    ndcg_p_unpop = min(ndcg_p, holdout_I - pop_items.shape[0])
    holdout_unpop = np.copy(holdout[~np.in1d(holdout[:, 1], pop_items),:])
    holdout_users_unpop = np.unique(holdout_unpop[:,0])

    pop_items_relabel = np.unique(convert_items[pop_items])
    holdout_relabel_unpop = np.copy(holdout_relabel[~np.in1d(holdout_relabel[:, 1], pop_items_relabel),:])
    holdout_users_relabel_unpop = np.unique(holdout_relabel_unpop[:,0])

    recommendations_unpop = ppm.recommend(topk,holdout_users_relabel_unpop.astype(int), excluded_items=pop_items_relabel).indices
    unpop_ndcg = ppm.nDCG(ndcg_p_unpop,holdout_users_relabel_unpop, holdout_relabel_unpop, excluded_items=pop_items_relabel)

    precision_unpop = np.zeros(holdout_users_relabel_unpop.shape[0])
    for en,user in enumerate(holdout_users_relabel_unpop):
        user_test_edges = holdout_relabel_unpop[holdout_relabel_unpop[:, 0] == user, 1]
        precision_unpop[en] = float(np.intersect1d(user_test_edges, recommendations_unpop[en, :]).shape[0]) / float(np.minimum(20., float(user_test_edges.shape[0])))


    ## Plotting
    _, lookup_user_degree = np.unique(lookup[:,0],return_counts=True)
    plt.clf()
    fig, [ax1,ax2,ax3,ax4] = plt.subplots(4, 1, sharex=True)
    plt.xscale('log')

    ax1.set_ylabel('precision')
    ax1.set_ylim([0,1])
    ax1.plot(lookup_user_degree[holdout_users].astype(np.int), precision, "o")

    ax2.set_ylabel("unpopular precision (degree < {})".format(thr))
    ax2.set_ylim([0,1])
    ax2.plot(lookup_user_degree[holdout_users_unpop].astype(np.int), precision_unpop, "o")


    ax3.set_xlabel('degree in test lookup')
    ax3.set_ylabel("ndcg (p={})".format(ndcg_p))
    ax3.set_ylim([0,1])
    ax3.plot(lookup_user_degree[holdout_users].astype(np.int), ndcg, "o")


    ax4.set_xlabel('degree in test lookup')
    ax4.set_ylabel("unpop ndcg (deg < {})".format(thr))
    ax4.set_ylim([0,1])
    ax4.plot(lookup_user_degree[holdout_users_unpop].astype(np.int), unpop_ndcg, "o")

    plt.savefig(data_dir+"/evaluation")

    print ("===========================================================================")
    print ("RUN SUMMARY")
    print ("===========================================================================")

    train_si,train_su = estimate_sigmas_det(train)
    train_size_i,train_size_u = (model.size_i,model.size_u) #estimate_size(train,tu, su, a, b,ti, si, c, d,K)
    lookup_si, lookup_su = estimate_sigmas_det(lookup)
    lookup_size_i, lookup_size_u = estimate_size(lookup,tu, su, a, b,ti, si, c, d,K)
    holdout_si, holdout_su = estimate_sigmas_det(holdout)
    holdout_size_i, holdout_size_u = (ppm.size_i,ppm.size_u)# estimate_size(ppm,tu, su, a, b,ti, si, c, d,K)
    post_sample_train_si, post_sample_train_su = estimate_sigmas_det(post_sample_train)
    post_sample_train_size_i, post_sample_train_size_u = estimate_size(post_sample_train,tu, hparams['su'], a, b,ti, hparams['si'], c, d,K)
    post_sample_holdout_si, post_sample_holdout_su = estimate_sigmas_det(post_sample_holdout)
    post_sample_holdout_size_i, post_sample_holdout_size_u = estimate_size(post_sample_holdout,tu, hparams['su'], a, b,ti, hparams['si'], c, d,K)

    ## Summary 
    # Dataset
    print("\n")
    print("Dataset Summary")
    print("Train Data set: # edges: {}, # users: {}, # items: {}".format(train.shape[0], U, I))
    print("Lookup Data set: # edges: {}, # users: {}, # items: {}".format(lookup.shape[0], lookup_U, lookup_I))
    print("Holdout Data set: # edges: {}, # users: {}, # items: {}".format(holdout.shape[0], holdout_U, holdout_I))
    print("\n")

    # Recommendation Metrics
    print("Recommendation Evaluation Summary")
    print("Mean Precision: {}".format(np.mean(precision)))
    # print("Std Precision: {}".format(np.std(precision)))
    print("Mean unpop Precision: {}".format(np.mean(precision_unpop)))
    # print("Std unpop Precision: {}".format(np.std(precision_unpop)))
    print("Mean nDCG: {}".format(np.mean(ndcg)))
    # print("Std nDCG: {}".format(np.std(ndcg)))
    print("Mean unpop nDCG: {}".format(np.mean(unpop_ndcg)))
    # print("Std nDCG: {}".format(np.std(ndcg)))
    print("\n")

    # Posterior samples
    print("Posterior Samples Summary")
    print("Posterior samples Train: # edges: {}, # users: {}, # items: {}".format(post_sample_train.shape[0], \
                        np.unique(post_sample_train[:,0]).shape[0],np.unique(post_sample_train[:,1]).shape[0]))
    print("Posterior samples Holdout: # edges: {}, # users: {}, # items: {}".format(post_sample_holdout.shape[0], \
                        np.unique(post_sample_holdout[:,0]).shape[0],np.unique(post_sample_holdout[:,1]).shape[0]))
    print("\n")

    # Estimators
    print("Estimators")
    print("Train true: sigma_u = {}, sigma_i = {}, size_u = {}, size_i = {}".format(su, si, size_u*0.8, size_i))
    print("Train estimated: sigma_u = {}, sigma_i = {}".format(train_su,train_si))
    print("Train estimated (assuming sigma_u = {},sigma_i = {}): size_u = {}, size_i: {}".format(hparams['su'],hparams['si'],train_size_u,train_size_i))
    print("Post Train estimated: sigma_u = {}, sigma_i = {}".format(post_sample_train_su,post_sample_train_si))
    print("Post Train estimated (assuming sigma_u = {}, sigma_i={}): size_u = {}, size_i = {}".format(hparams['su'],hparams['si'],post_sample_train_size_u,post_sample_train_size_i))
    print("Lookup true: sigma_u: {}, sigma_i: {}, size_u: {}, size_i: {}".format(su, si, size_u*0.2, size_i*0.8))
    print("Lookup estimated: sigma_u = {}, sigma_i = {}".format(lookup_su, lookup_si))
    print("Lookup estimated (assuming sigma_u = {}, sigma_i = {}): size_u = {}, size_i = {}".format(hparams['su'], hparams['si'], lookup_size_u, lookup_size_i))
    print("Holdout true: sigma_u = {}, sigma_i = {}, size_u = {}, size_i = {}".format(su, si, size_u*0.2, size_i*0.2))
    print("Holdout estimated: sigma_u = {}, sigma_i = {}".format(holdout_su, holdout_si))
    print("Holdout estimated (assuming sigma_u = {}, sigma_i = {}): size_u = {}, size_i = {}".format(hparams['su'], hparams['si'], holdout_size_u, holdout_size_i))
    print("Post Holdout estimated: sigma_u = {}, sigma_i = {}".format(post_sample_holdout_su,post_sample_holdout_si,post_sample_holdout_size_u,post_sample_holdout_size_i))
    print("Post Holdout estimated (assuming sigma_u = {}, sigma_i = {}): size_u = {}, size_i = {}".format(hparams['su'],hparams['si'],post_sample_holdout_size_u,post_sample_holdout_size_i))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("dir", help="directory containing the data file", type=str)

    # Optional arguments
    parser.add_argument("-tu", "--tu", type=float, help="default tu = 1.", default=1.)
    parser.add_argument("-su", "--su", type=float, help="User sparsity parameter: For dense model, default su = -0.1; for sparse model, estimate is used", default = -0.1)
    parser.add_argument("-a", "--a", type=float, help="default a = 0.1", default = 0.1)
    parser.add_argument("-b", "--b", type=float, help="default b = 0.1", default = 0.1)
    parser.add_argument("-ti", "--ti", type=float, help="default ti = 1.", default = 1.)
    parser.add_argument("-si", "--si", type=float, help="Item sparsity parameter: For dense model, default si = -0.1; For sparse model, estimate is used", default = -0.1)
    parser.add_argument("-c", "--c", type=float, help="default c = 0.1", default = 0.1)
    parser.add_argument("-d", "--d", type=float, help="default d = 0.1", default = 0.1)
    parser.add_argument("-K", "--K", type=int, help="Number of communities: default K = 30", default = 30)
    parser.add_argument("-itr", "--itr", type=int, help="Number of iterations; default itr = 400", default = 400)
    parser.add_argument("-unpop", "--unpop", type=int, help="Percentage for unpopular movies; default unpop = 0.95", default = 0.95)
    parser.add_argument("-seed", "--seed", type=int, help="default seed = None", default=None)
    parser.add_argument("-topK", "--topK", type=int, help="Top K recommendation: default topK = 10", default = 10)
    parser.add_argument("-nDCG", "--nDCG", type=int, help="Number of recommendations to rank for Normalized Discounted Cumulative Gain: default nDCG = 1000", default = 1000)

    parser.add_argument("-mg", "--multi_graph", action="store_true", 
                        help="Run Multi Graph mode; Default is simple graph mode")
    parser.add_argument("-ncr", "--ncr", action="store_true", 
                        help="Do not compute remainder mass")
    parser.add_argument("-dense", "--dense", action="store_true", 
                        help="Run dense model: Need to set si and su (default values will be used if not specified)")
    parser.add_argument("-gpu", "--gpu", action="store_true", 
                        help="Run with the GPU")
    args = parser.parse_args()
    hparams = {'tu': args.tu, 'su': args.su, 'a': args.a, 'b': args.b, 
               'ti': args.tu, 'si': args.su, 'c': args.c, 'd': args.d,
               'K': args.K, 'itr': args.itr, 
               'unpop':args.unpop, 'topK':args.topK, 'nDCG':args.nDCG}
    
    hparams['sparse'] = not(args.dense)

    data_dir = args.dir
    print "Loading train set"
    with open(data_dir+"/train.pkl", "rb") as f:
        train = pickle.load(f)

    print train.shape
    print "Loading test_lookup set"

    with open(data_dir+"/test_lookup.pkl", "rb") as f:
        lookup = pickle.load(f)

    print "Loading test_holdout set"
    with open(data_dir+"/test_holdout.pkl", "rb") as f:
        holdout = pickle.load(f)

    if args.dense:
        hparams['size_i'] = 0.
        hparams['size_u'] = 0.
    else: 
        hparams['si'], hparams['su'] =  estimate_sigmas_det(train)
        hparams['size_i'], hparams['size_u'] = estimate_size(train, hparams['tu'], hparams['su'], hparams['a'], hparams['b'],
                                                            hparams['ti'], hparams['si'], hparams['c'], hparams['d'],
                                                            hparams['K'])

    print hparams
    main(train, lookup, holdout, hparams, args.gpu, not(args.multi_graph), data_dir, hparams['sparse'], args.seed)
    pass
