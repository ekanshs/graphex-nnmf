import sys
sys.path.append('../')

import argparse
import pickle
import numpy as np
import tensorflow as tf

from p_sampling import user_p_sample
from data.data_splitting import clean_p_samp_split, clean_item_p_sample
from data.sim_data import sim_data

def main(data_dir, params):
    # Store true_gam, true_theta, true_beta
    # Store edges to data.txt
    print "Simulating data with following paramaters"
    print params
    [true_gam, true_theta, true_omega, true_beta, edges] = sim_data(params['tu'], params['su'], params['size_u'], params['a'], params['b'],
                                                                    params['ti'], params['si'], params['size_i'], params['c'], params['d'],
                                                                    params['K'], params['eps'])

    print edges[:,2].mean()

    print("Number of occupied pairs in the dataset: {}".format(edges.shape[0]))
    print("Number of users in the dataset: {}".format(np.unique(edges[:,0]).shape[0]))
    print("Number of items in the dataset: {}".format(np.unique(edges[:,1]).shape[0]))
    print("e / (U*I): {}").format(np.float(edges.shape[0]) / np.float(np.unique(edges[:,0]).shape[0] * np.unique(edges[:,1]).shape[0]))
    print ("e / (size_u*size_i): {}").format(np.float(edges.shape[0]) / np.float(params['size_u']*params['size_i']))

    print "Storing data at: " + data_dir + "/data.pkl"
    with open(data_dir+"/data.pkl", "wb") as f:
        pickle.dump(edges, f)
        pickle.dump(true_gam, f)
        pickle.dump(true_theta, f)
        pickle.dump(true_omega, f)
        pickle.dump(true_beta, f)
        pickle.dump(params, f)
    pass

    print "Splitting the dataset"
    edges_train, edges_test = user_p_sample(edges, p=0.8)
    edges_train, edges_test, allusers_train, allitems = clean_p_samp_split(edges_train, edges_test)
    true_gam_train = true_gam[allusers_train]
    true_theta_train = true_theta[allusers_train,:]
    true_omega_train = true_omega[allitems]
    true_beta_train = true_beta[allitems,:]

    print("Edges in train set: {}".format(edges_train.shape[0]))
    print("Edges in test set: {}".format(edges_test.shape[0]))
    print("Storing train set to {}".format(data_dir+"/train.pkl"))
    with open(data_dir+"/train.pkl", "wb") as f:
        pickle.dump(edges_train, f)
        pickle.dump(true_gam_train, f)
        pickle.dump(true_theta_train, f)
        pickle.dump(true_omega_train, f)
        pickle.dump(true_beta_train, f)
        pickle.dump(params, f)

    # Split edges_test to test_holdout and test_look
    print "Splitting the test set to lookup and holdout"
    edges_test_lookup, edges_test_holdout = clean_item_p_sample(edges_test, 0.8)
    print("Storing test lookup set to {}".format(data_dir+"/test_lookup.pkl"))
    print("Edges in test lookup set: {}".format(edges_test_lookup.shape[0]))
    print("Edges in test holdout set: {}".format(edges_test_holdout.shape[0]))
    with open(data_dir+"/test_lookup.pkl", "wb") as f:
        pickle.dump(edges_test_lookup, f)
    print("Storing test holdout set to {}".format(data_dir+"/test_holdout.pkl"))
    
    with open(data_dir+"/test_holdout.pkl", "wb") as f:
        pickle.dump(edges_test_holdout, f)
    print("Done!")

if __name__ == '__main__':
    ## Setup argument parser
    parser = argparse.ArgumentParser()

    ## Optional arguments
    parser.add_argument("-tu", "--tu", type=float, help="default tu = 1.", default=1.0)
    parser.add_argument("-su", "--su", type=float, help="default su = 0.2", default=0.2)
    parser.add_argument("-size_u", "--size_u", type=float, help="default size_u = 100", default=100)
    parser.add_argument("-a", "--a", type=float, help="default a = 0.1", default=0.1)
    parser.add_argument("-b", "--b", type=float, help="default b = 0.1", default=0.1)
    
    parser.add_argument("-ti", "--ti", type=float, help="default ti = 1.", default=1.0)
    parser.add_argument("-si", "--si", type=float, help="default si = 0.2", default=0.2)
    parser.add_argument("-size_i", "--size_i", type=float, help="default size_i = 100", default=100)
    parser.add_argument("-c", "--c", type=float, help="default c = 0.1", default=0.1)
    parser.add_argument("-d", "--d", type=float, help="default d = 0.1", default=0.1)
    
    parser.add_argument("-K", "--K", type=int, help="default K = 30", default=30)
    parser.add_argument("-eps", "--eps", type=float, help="default eps = 1e-6", default=1e-6)
    parser.add_argument("-p", "--p", type=float, help="default p = 0.8", default=0.8)

    ## Positional arguments
    parser.add_argument("dir", help="directory to store the data file", type=str)
    args = parser.parse_args()

    ## Set to parsed or default value
    params = {'tu': args.tu, 'su': args.su, 'a': args.a, 'b': args.b, 'size_u': args.size_u,
              'ti': args.ti, 'si': args.si, 'c': args.c, 'd': args.d, 'size_i': args.size_i,
              'K': args.K, 'eps': args.eps, 'p': args.p}

    main(args.dir, params)
