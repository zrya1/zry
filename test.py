

import torch.nn.functional

a=torch.rand((1,2,3))
print(a)

b=a.data.new([1]).view(1,1)
print(b)