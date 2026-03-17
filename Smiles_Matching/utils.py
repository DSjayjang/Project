import numpy as np
import pandas as pd
import time

def kl_divergence(p, q):
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

def sqrt_js_divergence(p, q):
    p = p.astype(float)
    q = q.astype(float)
    
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    js = max(js, 0.0)
    return np.sqrt(js)