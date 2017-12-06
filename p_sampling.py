import numpy as np


def user_p_sample(graph, p):
    """
    :param graph: nparray of (multi-)edges, with graph[0] = [user_idx, item_idx, edge_val]
    :param p: train set is p-sampling of users of graph
    :return: train, test
    """
    users = np.unique(graph[:, 0])
    U = users.shape[0]
    pick_number = np.random.binomial(U, p)
    train_users = np.random.choice(users, pick_number, replace=False)

    in_train = np.in1d(graph[:,0], train_users)
    in_test = np.invert(in_train)
    return np.copy(graph[in_train]), np.copy(graph[in_test])

def item_p_sample(graph, p):
    """
    :param graph: nparray of (multi-)edges, with graph[0] = [user_idx, item_idx, edge_val]
    :param p: train set is p-sampling of users of graph
    :return: train, test
    """
    g = graph.copy()
    g = g[:,[1,0]]
    train, test = user_p_sample(g, p)
    return train[:,[1,0]], test[:,[1,0]]