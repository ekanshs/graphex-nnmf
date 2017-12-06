import numpy as np
from scipy.special import gamma as gamfn


def sample_ggp(size, sig, tau, eps):
    """
    Sample weights from truncated GGP.
    Equiv, sample from Poisson process w mean measure size*(1/gamma(1-sig))w^-(1+sig)*exp(-tau*w)1[w>eps]
    This is the adaptive thinning scheme described in Appendix C of Todeschini & Caron
    :param size:
    :param sig:
    :param tau:
    :param eps:
    :return:
    """

    if sig < 0:
        # took this from Francois Caron's matlab implementation
        rate = size / (-sig) * np.power(tau,sig)
        K = np.random.poisson(rate)
        N = np.random.gamma(-sig, 1 / tau, K)
        return np.sort(N[N >= eps]).astype(np.float32)

    assert (tau > 0 and 0 < sig < 1) or (tau > 0 and -1 < sig <= 0)

    t= eps

    N = []
    while True:
        r = np.random.exponential(1)
        if np.log(r) > _log_Gt(np.inf, t, size, sig, tau):
            break
        else:
            tn = _Gt_inv(r, t, size, sig, tau)

        log_odds = _log_rho_trunc(tn, size, sig, tau, eps)-_log_gt(tn, t, size, sig, tau)
        if log_odds > np.log(1):
            N.append(tn)
        elif log_odds >= -40:
            p = np.exp(log_odds)
            if np.random.binomial(1,p):
                N.append(tn)

        t = tn

    return np.array(N, dtype=np.float32)


def _log_Gt(s, t, size, sig, tau):
    if tau > 0:
        return np.log(size / gamfn(1.-sig)) - np.log(tau) + (-1-sig)*np.log(t) + np.log((np.exp(-tau*t)-np.exp(-tau*s)))
    else:
        return np.log(size / gamfn(1.-sig))- np.log(sig) + np.log(np.power(t,-sig) - np.power(s,-sig))


def _Gt_inv(r, t, size, sig, tau):
    if tau > 0:
        return t - 1. / tau * np.log(1. - r * tau * gamfn(1.-sig) / (size * np.power(t, -1.-sig)*np.exp(-t*tau)))
    else:
        return np.power(np.power(t,-sig) - np.divide(r*sig*gamfn(1.-sig),size), -np.divide(1,sig))


def _log_rho_trunc(w, size, sig, tau, eps):
    if w > eps:
        return np.log(size)-np.log(gamfn(1.-sig)) + (-1-sig)*np.log(w) - tau*w
    else:
        return -np.inf


def _log_gt(s, t, size, sig, tau):
    if tau > 0:
        return np.log(size)-np.log(gamfn(1.-sig)) + (-1-sig)*np.log(t) - tau*s
    else:
        return np.log(size)-np.log(gamfn(1.-sig)) + (-1-sig)*np.log(s)




def main():
    size = 100.
    sig = 0.25
    tau = 0.1
    eps = 0.00001

    g_samp = sample_ggp(size, sig, tau, eps)
    print(g_samp)
    print(np.sum(g_samp))
    print(np.sum(g_samp[g_samp<0.001]))

if __name__ == "__main__":
    main()

