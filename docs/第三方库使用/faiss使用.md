# faiss使用

https://github.com/facebookresearch/faiss/wiki/Getting-started



## lsh search

```python
mport numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
nlist = 100
m = 8
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss                   # make faiss available
index = faiss.IndexLSH(d,4*8*d)
#assert not index.is_trained
print(index.is_trained)
index.train(xb)
assert index.is_trained
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xq, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
```

