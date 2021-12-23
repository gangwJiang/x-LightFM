import numpy as np
import time 
import torch
a = np.random.randint(0,2,(1000000,4))
b = np.zeros_like(a)
# a = torch.from_numpy(a).to(4)
begin= time.time()
for j in range(b.shape[1]):
    cnt = 0
    for i in range(a.shape[0]):
        if a[i,j] == 1:
            b[i,j] = cnt
            cnt+=1    

print(time.time()-begin)