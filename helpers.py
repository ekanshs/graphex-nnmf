import numpy as np
import tensorflow as tf

TINY = tf.constant(1e-7)


def to_prob(unnorm):
    """
    :param unnorm: tensor with non-negative entries
    :return: tensor; probability distribution given by normalizing the input on its last dimension
    """
    stable = unnorm + TINY
    return tf.realdiv(stable, tf.reduce_sum(stable, -1, keep_dims=True))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def zero_index(graph, axis=0, return_inverse=True):
    """
    Zero index the given axis
    the mapping is label -> rank(label), e.g. [0, 7, 7, 4, 8] -> [0, 2, 2, 1, 3]        
    """
    column = np.unique(graph[:,axis])
    num_unique = column.shape[0]
    graph_relabel = graph.copy()
    convert = np.zeros(np.max(column)+1, dtype=np.int32)
    # critical fact: lookup_relabel[j,1] = k means that this item corresponds to lookup_items[k] in the original labeling
    convert[column] = np.r_[0:num_unique]
    graph_relabel[:,axis] = convert[graph[:,axis]]
    if return_inverse:
        return [graph_relabel, convert]
    else:
        return graph

# from comment at https://github.com/tensorflow/tensorflow/issues/206
def gather_along_second_axis(data, indices):
    # tf.gather along second axis
    indices32 = tf.cast(indices, tf.int32)
    flat_indices = tf.tile(indices32[None, :], [tf.shape(data)[0], 1])
    batch_offset = tf.range(0, tf.shape(data)[0]) * tf.shape(data)[1]
    flat_indices = tf.reshape(flat_indices + batch_offset[:, None], [-1])
    flat_data = tf.reshape(data, tf.concat([[-1], tf.shape(data)[2:]], 0))
    result_shape = tf.concat([[tf.shape(data)[0], -1], tf.shape(data)[2:]], 0)
    result = tf.reshape(tf.gather(flat_data, flat_indices), result_shape)
    shape = data.shape[:1].concatenate(indices32.shape[:1])
    result.set_shape(shape.concatenate(data.shape[2:]))
    return result


def compute_degrees(edge_vals, edge_idx, U, I):
    """
    Takes a weighted graph specified as an edge list and computes the user and item degrees
    :param edge_vals:
    :param edge_idx:
    :param U:
    :param I:
    :return:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                """
    # reasonable alternative:
    # possibly better alternative
    # create variable to hold the degree bits, and use https://www.tensorflow.org/api_docs/python/tf/scatter_add
    adj_mat = tf.SparseTensor(edge_idx, edge_vals, [U, I])
    return tf.sparse_reduce_sum(adj_mat, axis=1), tf.sparse_reduce_sum(adj_mat, axis=0)


def compute_degrees2(edge_vals, edge_idx, U, I):
    """
    Takes a weighted graph specified as an edge list and computes the weighted user and item degrees
    Actually somewhat broader: if edge_vals has shape [e, k] then we produce [u,k] and [i,k] values corresponding to the
     weighted degrees in each of the k graphs

    WARNING: this relies on undocumented behaviour of tf.scatter_nd, and may well break going forward (works as of tf1.3)

    :param edge_vals: shape [edges, K]
    :param edge_idx: shape [edges]
    :return user_degree, item_degree: shape [U,K] and [I,K] respectively
    """

    K = edge_vals.shape.as_list()[1]

    # scatter_nd (as of version 1.1) adds duplicate indices
    user_degree = tf.scatter_nd(tf.expand_dims(edge_idx[:,0], axis=1), edge_vals, [U,K])
    item_degree = tf.scatter_nd(tf.expand_dims(edge_idx[:,1], axis=1), edge_vals, [I,K])

    return user_degree, item_degree


def assign_list(vars, new_values):
    """
    assigns each element of vars the value in new_values
    :param vars: list of tf variables
    :param new_values: list of tensors
    """
    # warning: zip works differently in Python 2 and 3
    return [old_var.assign(new_val) for (old_var, new_val) in zip(vars, new_values)]


def gather_and_add(param_list, index_list):
    with tf.name_scope("gather_and_add"):
        gathered = [tf.gather(param, indices) for (param, indices) in zip(param_list, index_list)]
        return tf.add_n(gathered)


def tensor_split(ary, num_splits, axis=0):
    """
    tf.split when num_splits doesn't evenly divide array.shape[axis]
    mimics
    :param ary:
    :param num_splits:
    :param axis:
    :return:
    """
    len = ary.shape.as_list()[axis]
    _num_splits = min(num_splits, ary.shape[axis].value)

    split = np.repeat(np.floor(len / _num_splits), _num_splits).astype(np.int)
    rem = np.mod(len, _num_splits)
    split[:rem]+=1
    return tf.split(ary, split, axis=axis)


def setdiffRows(A, B, assume_unique=False):
    """
    Compute the rows of A that are not in B
    :param A: np.array
    :param B: np.array
    :return: rows of A not in B
    """
    AA = A.copy()
    BB = B.copy()

    nrows, ncols = AA.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [AA.dtype]}
    C = np.setdiff1d(AA.view(dtype), BB.view(dtype), assume_unique=assume_unique)
    return C.view(AA.dtype).reshape(-1, ncols)
