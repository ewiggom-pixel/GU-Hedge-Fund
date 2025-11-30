import numpy as np

def kalman_filter_dlm(y, F, alpha, Q, R, beta0, P0):
    y = np.asarray(y, dtype=float).reshape(-1)
    F = np.asarray(F, dtype=float)
    T, K = F.shape
    m_pred = np.zeros((T, K))
    P_pred = np.zeros((T, K, K))
    m_filt = np.zeros((T, K))
    P_filt = np.zeros((T, K, K))
    loglike = 0.0
    m_prev = np.asarray(beta0, dtype=float).reshape(K)
    P_prev = np.asarray(P0, dtype=float).reshape(K, K)
    I = np.eye(K)
    for t in range(T):
        ft = F[t]
        m_prior = m_prev
        P_prior = P_prev + Q
        y_hat = alpha + ft @ m_prior
        nu = y[t] - y_hat
        S = ft @ P_prior @ ft + R
        Kt = P_prior @ ft / S
        m_post = m_prior + Kt * nu
        KH = np.outer(Kt, ft)
        P_post = (I @ P_prior) - KH @ P_prior
        m_pred[t] = m_prior
        P_pred[t] = P_prior
        m_filt[t] = m_post
        P_filt[t] = P_post
        loglike += -0.5 * (np.log(2 * np.pi) + np.log(S) + (nu ** 2) / S)
        m_prev = m_post
        P_prev = P_post
    return m_pred, P_pred, m_filt, P_filt, loglike

def rts_smoother_dlm(m_pred, P_pred, m_filt, P_filt):
    m_pred = np.asarray(m_pred, dtype=float)
    P_pred = np.asarray(P_pred, dtype=float)
    m_filt = np.asarray(m_filt, dtype=float)
    P_filt = np.asarray(P_filt, dtype=float)
    T, K = m_filt.shape
    m_smooth = np.zeros_like(m_filt)
    P_smooth = np.zeros_like(P_filt)
    C_smooth = np.zeros_like(P_filt)
    m_smooth[-1] = m_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for t in range(T - 2, -1, -1):
        P_prior_next = P_pred[t + 1]
        A = P_filt[t] @ np.linalg.inv(P_prior_next)
        m_smooth[t] = m_filt[t] + A @ (m_smooth[t + 1] - m_pred[t + 1])
        P_smooth[t] = P_filt[t] + A @ (P_smooth[t + 1] - P_prior_next) @ A.T
        C_smooth[t] = A @ P_smooth[t + 1]
    return m_smooth, P_smooth, C_smooth

def em_dlm_multifactor(y, F, max_iter=50, tol=1e-5):
    y = np.asarray(y, dtype=float).reshape(-1)
    F = np.asarray(F, dtype=float)
    T, K = F.shape
    X = np.column_stack([np.ones(T), F])
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = float(beta_ols[0])
    beta0 = beta_ols[1:].copy()
    resid = y - X @ beta_ols
    R = float(np.var(resid, ddof=1))
    Q = 1e-6 * np.eye(K)
    P0 = 1.0 * np.eye(K)
    prev_ll = -np.inf
    for _ in range(max_iter):
        m_pred, P_pred, m_filt, P_filt, ll = kalman_filter_dlm(
            y, F, alpha, Q, R, beta0, P0
        )
        m_smooth, P_smooth, C_smooth = rts_smoother_dlm(
            m_pred, P_pred, m_filt, P_filt
        )
        alpha_new = np.mean(y - np.sum(F * m_smooth, axis=1))
        R_num = 0.0
        for t in range(T):
            ft = F[t]
            mt = m_smooth[t]
            Pt = P_smooth[t]
            err_mean = y[t] - alpha_new - ft @ mt
            R_num += err_mean**2 + ft @ Pt @ ft
        R_new = R_num / T
        Q_num = np.zeros((K, K))
        for t in range(1, T):
            mt = m_smooth[t]
            mt1 = m_smooth[t - 1]
            Pt = P_smooth[t]
            Pt1 = P_smooth[t - 1]
            Ct1 = C_smooth[t - 1]
            Mt = Pt + np.outer(mt, mt)
            Mt1 = Pt1 + np.outer(mt1, mt1)
            Mt_t1 = Ct1 + np.outer(mt, mt1)
            Q_num += Mt + Mt1 - Mt_t1 - Mt_t1.T
        Q_new = Q_num / (T - 1)
        Q_new = 0.5 * (Q_new + Q_new.T)
        alpha, R, Q = alpha_new, R_new, Q_new
        if np.abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return {
        "alpha": alpha,
        "Q": Q,
        "R": R,
        "beta0": beta0,
        "P0": P0,
        "beta_smooth": m_smooth,
        "P_smooth": P_smooth,
        "loglike": prev_ll,
    }
