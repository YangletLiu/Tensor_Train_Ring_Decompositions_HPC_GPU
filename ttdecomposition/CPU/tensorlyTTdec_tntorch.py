import numpy as np
import time
import tntorch as tnt
import torch
from tensorly.decomposition import matrix_product_state
from tensorly.contrib.decomposition import matrix_product_state_cross


def metrics(t2, full):
    print(t2)
    print('Compression ratio: {}/{} = {:g}'.format(full.numel(), t2.numel(), full.numel() / t2.numel()))
    print('Relative error:', tn.relative_error(full, 2))
    print('RMSE:', tn.rmse(full, t2))
    print('R^2:', tn.r_squared(full, t2))


k = 300
a = np.random.rand(k, k)
a = a.astype(np.float32)
b = np.random.rand(k, k)
b = b.astype(np.float32)
c = np.random.rand(k, k*k)
c = c.astype(np.float32)
h_X = np.dot(c, (np.kron(a, b).transpose()))

X = torch.Tensor(h_X.reshape(k, k, k))
print("start TT decomposition")
a = time.time()
for i in range(0, 5):
    t = tnt.Tensor(X)
    t.round_tt(eps=1e-6)
    #factors = matrix_product_state_cross(X, [1, 120, 120, 1])
b = time.time()
print("time:", (b-a)/5.0)
metrics(t, X)



print(len(factors))
print([f.shape for f in factors])