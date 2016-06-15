import numpy as np
import theano.tensor as T
from theano import *

f = T.vector("f")
W = T.matrix("W")

numerator = T.exp(T.dot(W, f))
z = numerator / T.sum(numerator)

log_reg = function([W, f], [z])

test_w = np.random.rand(10, 20)
feat_vec = np.random.rand(20)

print np.sum(log_reg(test_w, feat_vec))

