import numpy as np
from p_sampling import user_p_sample



def clean_item_p_sample(graph, p):
    """
    p sample the items to create a train-test split, and then clean up the resulting test set so that
    test contains only users that are also in train
    Note that we *do not* zero index the items (because these will be passed in to something that contains the full item set)
    :param graph:
    :param p: train set is p-sampling of items of graph
    :return:
    """
    lazy = np.copy(graph)
    # interchange users and items
    lazy[:,0] = graph[:,1]
    lazy[:,1] = graph[:,0]

    ltrain, ltest = user_p_sample(lazy, p)
    # eliminate any users in test that aren't also in train, and then give those users a new common zero index
    # do not reindex the items! (that would break prediction code)
    ltrain, ltest = clean_p_samp_split(ltrain, ltest, zi_train_u=False, zi_test_u=False)

    train = ltrain.copy()
    train[:,0] = ltrain[:,1]
    train[:,1] = ltrain[:,0]

    test = ltest.copy()
    test[:,0] = ltest[:,1]
    test[:,1] = ltest[:,0]

    return train, test


def clean_p_samp_split(train, test, zi_items=True, zi_train_u=True, zi_test_u=True):
    """
    Zero index + remove test edges with movies that are not in train

    Return a version of test that contains only items that are also in train
    (to avoid trying to make predictions on unknown items)
    and also relabel the users and items st the sets are contiguous (to avoid degree 0 vertices)

    :param train: nparray of (multi-)edges, with train[0] = [user_idx, item_idx, edge_val]
    :param test: nparray of (multi-)edges, with test[0] = [user_idx, item_idx, edge_val]
    :return: cleaned test and train,
    and arrays giving the mapping of user and item indices in the cleaned set to indices in the set passed in
    """

    # Zero index Users in Train
    if zi_train_u:
        uids_train = train[:,0]
        allusers_train = np.unique(uids_train)
        numusers_train = allusers_train.size
        convertusers_train = np.zeros(np.max(allusers_train)+1, dtype=np.int64)
        convertusers_train[allusers_train] = np.r_[0:numusers_train]
        train[:,0] = convertusers_train[uids_train]

    # Zero index Users in Test
    if zi_test_u:
        uids_test = test[:,0]
        allusers_test = np.unique(uids_test)
        numusers_test = allusers_test.size
        convertusers_test = np.zeros(np.max(allusers_test)+1, dtype=np.int64)
        convertusers_test[allusers_test] = np.r_[0:numusers_test]
        test[:,0] = convertusers_test[uids_test]

    # Remove edges in test if item is not in train
    train_items = np.unique(train[:,1])
    test = test[np.in1d(test[:,1], train_items)]

    # Zero index items (iids: item ids)
    if zi_items:
        iids = train[:,1]
        allitems = np.unique(train[:,1])
        numitems = allitems.size
        convertitems = np.zeros(np.max(allitems)+1, dtype=np.int64)
        convertitems[allitems] = np.r_[0:numitems]
        train[:,1] = convertitems[iids]
        test[:,1] = convertitems[test[:,1]]

    if zi_items and zi_train_u:
        # return allusers_train and allitems to allow us to track the ground truth params corresponding to the training set
        return train, test, allusers_train, allitems
    else:
        return train, test

def naive_train_test_split(data, split_p=0.2, users_p=0.6):
    """
    Naive train-test splitting: 
    Idea is to select some percentage of users (user_p) and select a percentage of their edges (split_p).

    split_p: 
    user_p: 
    """
    n_edges = data.shape[0]
    users = np.unique(data[:,0])
    n_users = users.shape[0]
    n_items = np.unique(data[:,1]).shape[0]    
    train_set = data

    test_set = np.empty((n_edges,data.shape[1]), int)
    # select user_p fraction of the users
    selected_users = np.random.choice(n_users, int(users_p*n_users), replace=False)
    curr_test = 0
    # per user select split_p fraction of edges st. no zero degree items
    for en, user in enumerate(selected_users):
        print user
        train_mask = train_set[:,0]==user
        user_indices = np.array(train_mask.nonzero())[0]
        n_user_edges = user_indices.shape[0]
        n_user_train = np.maximum(np.random.binomial(n_user_edges, 1.-split_p), 1)
        n_new_test = n_user_edges - n_user_train
        user_train = user_indices[np.random.choice(n_user_edges, n_user_train, replace=False)]
        train_mask[user_train] = False
        test_set[curr_test:curr_test+n_new_test] = train_set[train_mask]
        curr_test = curr_test + n_new_test
        train_set = train_set[~train_mask]
    test_set = test_set[:curr_test]
    # make sure no degree zero items
    # unique items in test set not in train
    test_items = np.unique(test_set[:,1])
    train_items = np.unique(train_set[:,1])
    items_not_in_train = np.setdiff1d(test_items, train_items)
    for en, item in enumerate(items_not_in_train):
        indices = np.array((test_set[:,1]==item).nonzero())[0]
        # select an index at random
        selected_edge_idx = indices[np.random.choice(indices.shape[0])]
        # insert the edges in train
        train_set = np.concatenate((train_set, test_set[selected_edge_idx].reshape(1,-1)),axis=0)
        # remove the edge from test
        test_set = np.delete(test_set, selected_edge_idx, 0)

    return train_set, test_set
