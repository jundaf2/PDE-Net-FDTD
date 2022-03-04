from numpy import *
from scipy.special import factorial
import torch
from utils import tensordot
def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim()-1
    for i in range(k):
        x = tensordot(mats[k-i-1], x, dim=[1,k])
    x = x.permute([k,]+list(range(k))).contiguous() #最后一列换到第一列
    x = x.view(sizex) #保持原来形状
    return x

def flat_order_bank(_order_bank):
    for a in _order_bank:
        for o in a:
            yield o
def _inv_equal_order_m(d,m):
    A = []
    assert d >= 1 and m >= 0
    if d == 1:
        A = [[m,],]
        return A
    if m == 0:
        for i in range(d):
            A.append(0)
        return [A,]
    for k in range(m+1):
        B = _inv_equal_order_m(d-1,m-k)
        for b in B:
            b.append(k)
        A = A+B
    return A
def _torch_setter_by_index(t,i,v): # tensor, index list, int
    for j in i[:-1]:
        t = t[j]
    t[i[-1]] = v

def _less_order_m(d,m):
    A = []
    n_e_o =[]
    num_each_order = 0
    for k in range(m+1):
        num_each_order+=1
        B = _inv_equal_order_m(d,k)
        for b in B:
            b.reverse()
        B.sort()
        B.reverse()
        A.append(B)
    return A

class _MK(torch.nn.Module):
    def __init__(self, shape): # shape is kernal size
        super(_MK, self).__init__()
        self._size = torch.Size(shape) # tensor的大小
        self._dim = len(shape) #维度
        M = []
        invM = []
        assert len(shape) > 0
        j = 0 # M 的编号
        for l in shape:
            M.append(zeros((l,l))) # 由每个维度大小单独拓展为正方形（l*l）的tensor
            for i in range(l): #最新的这个l*l tensor的第i行
                M[-1][i] = ((arange(l)-(l-1)//2)**i)/factorial(i)  #以(l-1)//2为对称轴的长度为l的多项式
            invM.append(linalg.inv(M[-1])) #最新正方形（l*l）的tensor的逆tensor
            # pytorch一般情况下，是将网络中的参数保存成orderedDict形式的，
            # 这里的参数其实包含两种，一种是模型中各种module含的参数，即nn.Parameter,
            # 我们当然可以在网络中定义其他的nn.Parameter参数，另一种就是buffer,
            # 前者每次optim.step会得到更新，而不会更新后者。
            self.register_buffer('_M'+str(j), torch.from_numpy(M[-1])) #很多个M
            self.register_buffer('_invM'+str(j), torch.from_numpy(invM[-1]))#很多个M的逆
            j += 1

    @property
    def M(self):
        return list(self._buffers['_M'+str(j)] for j in range(self.dim()))
    @property
    def invM(self):
        return list(self._buffers['_invM'+str(j)] for j in range(self.dim()))
    def size(self):
        return self._size
    def dim(self):
        return self._dim
    def _packdim(self, x): # x is moment
        assert x.dim() >= self.dim() #  not moment > kernal  self.dim()是moment的维度
        if x.dim() == self.dim(): #待pack的和当前dim的维度相等
            x = x[newaxis,:] #第一个维度之前再加一个维度
        x = x.contiguous()
        x = x.view([-1,]+list(x.size()[-self.dim():]))
        #除了第一个维度，其他维度的size为自身moment的所有维度对应的size
        return x

    def forward(self, m):
        sizem = m.size()
        m = self._packdim(m)
        m = _apply_axis_left_dot(m, self.invM)
        m = m.view(sizem)
        return m



d = 2
m = 3
s = 2*m-1
_order_bank = _less_order_m(d, m)
kernel_size = [s,]*d

N = 0
max_num_kernel_each_order=0
for a in _order_bank:
    N += len(a)
    max_num_kernel_each_order = len(a)
moment = torch.DoubleTensor(*([N,]+kernel_size)).zero_() #产生N个全零卷积核, Double类型的张量

for i, o in enumerate(flat_order_bank(_order_bank)):
    _torch_setter_by_index(moment[i], o, 1)

m2k = _MK(kernel_size)
kernel = m2k(moment)

def plt_kernels(kernel):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=m + 1, ncols=max_num_kernel_each_order, sharex='col',
                           sharey='row')
    axes = ax.flatten()
    i = 0
    for order, order_list in enumerate(_order_bank):
        for no_order_pair, order_pair in enumerate(order_list):
            i += 1
            axes[order * max_num_kernel_each_order + no_order_pair].matshow(kernel.detach().cpu().data[i - 1]
                                                                            ,cmap='RdBu')
            axes[order * max_num_kernel_each_order + no_order_pair].set_title(
                "$D{}{}$".format(order_pair[0], order_pair[1]))
    for i in axes:
        i.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

plt_kernels(kernel)