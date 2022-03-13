import torch

from wsl.modeling.roi_heads_nas.cell_operations_head import get_gumbels


def cnt(a, tau):
    s = torch.zeros_like(a)
    for i in range(10000):
        s += get_gumbels(a, tau)
    print(tau, s, torch.nn.functional.softmax(s / 10000, dim=1))


def cnt2(a, tau):
    s = torch.zeros_like(a)
    for i in range(10000):
        s += torch.nn.functional.gumbel_softmax(a, tau=tau, hard=True)
    print(tau, s, torch.nn.functional.softmax(s / 10000, dim=1))


a = torch.ones(1, 5)
a = a * 0.2

print(a)
print(torch.nn.functional.softmax(a, dim=1))
cnt(a, 10.0)
cnt(a, 1.0)
cnt(a, 0.1)

a[0, 0] = 0.2
a[0, 1] = 0.4
a[0, 2] = 0.8
a[0, 3] = 1.6
a[0, 4] = 3.2

print(a)
print(torch.nn.functional.softmax(a, dim=1))
cnt(a, 10.0)
cnt(a, 1.0)
cnt(a, 0.1)

print(a)
print(torch.nn.functional.softmax(a, dim=1))
cnt2(a, 10.0)
cnt2(a, 1.0)
cnt2(a, 0.1)
