import torch
import torch.nn as nn

from torch.nn import init
from torch.autograd import grad


c1 = nn.Linear(5, 10)
c2 = nn.Linear(10, 20)

inp = torch.rand(2,5)

o1 = c1(inp)
o2 = c2(o1)


points_grad = grad(
    outputs=o2,
    inputs=o1,
    grad_outputs=torch.ones_like(o2, requires_grad=False, device=o2.device),
    create_graph=True,
    retain_graph=True,
    only_inputs=True)[0]

gt = torch.ones(2, 10)
loss =nn.L1Loss()
import pdb
losss = loss(points_grad, gt)
pdb.set_trace()

losss.backward()
pdb.set_trace()


