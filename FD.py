"""Finite Difference"""
import numpy as np
from numpy import *
from numpy.linalg import *
from functools import reduce
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import MK
from utils import periodicpad

__all__ = ['MomentBank','FD1d','FD2d','FD3d']

def _inv_equal_order_m(d,m): # dim, max order, 按照固定参数生成大小为 (保证和为dim的排列组合数目, dim) 大小的矩阵
    # print("---------------------------")
    # print("m =", m, "d =", d)
    A = []
    assert d >= 1 and m >= 0
    if d == 1:  # 一维
        A = [[m,],]
        return A
    if m == 0:  # 0阶, dim=2二维用不上这个
        for i in range(d):  #
            # print(np.array(A).shape,"A =",A)
            A.append(0)
        return [A,]
    for k in range(m+1):
        B = _inv_equal_order_m(d-1,m-k) # B <- return A = [[m,],]
        for b in B:
            b.append(k) # [m,].append(0), [m-1,].append(1)
        A = A+B # []+[m,0]
        # print(np.array(A).shape," A =", np.array(A))
    return A

def _less_order_m(d,m):# dim, max order, 不好形容，很多个list，每个里面类似于_inv_equal_order_m
    # print("---------------------------")
    # print("m =", m, "d =", d)
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
        # print(np.array(A).shape, " A =", np.array(A))
    return A

# t是卷积核
def _torch_setter_by_index(t,i,v): # tensor, index list, int
    for j in i[:-1]:
        t = t[j]
    t[i[-1]] = v

def _torch_reader_by_index(t,i):
    for j in i:
        t = t[j]
        # print(t)
    return t

class MomentBank(nn.Module): #moment matrix(本身也是个kernel) for a given filter that will be used to constrain filters in the PDE-Net
    """
    generate moment matrix bank for differential kernels with order no more than max_order.
    Arguments:
        dim (int): dimension
        kernel_size (tuple of int): size of differential kernels
        max_order (int): max order of differential kernels
        dx (double): the MomentBank.kernel will automatically compute kernels
            according to MomentBank.moment and MomentBank.dx
        constraint (string): 'moment' or 'free'. Determine MomentBank.x_proj
            and MomentBank.grad_proj
    """
    def __init__(self, dim, kernel_size, max_order, dx=1.0, constraint='moment'):
        super(MomentBank, self).__init__()
        self._dim = dim
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size,]*self._dim # kernel_size*kernel_size*kernel_size...大小的kernel
        assert min(kernel_size) > max_order, print(min(kernel_size), max_order)  #根据卷积核构造定律而得
        self.m2k = MK.M2K(kernel_size)  #
        self._kernel_size = kernel_size.copy()
        self._max_order = max_order
        if not iterable(dx):
            dx = [dx,]*dim
        self._dx = dx.copy()
        self.constraint = constraint
        d = dim
        m = max_order
        self._order_bank = _less_order_m(d, m)  # 总阶数为 max_order，分别在不同的dim上施加不同order，是个约束化排列组合
        N = 0  # 总共 【order】对 的个数
        self.max_num_kernel_each_order = 0
        for a in self._order_bank:
            N += len(a)
            self.max_num_kernel_each_order = len(a)
        moment = torch.DoubleTensor(*([N,]+kernel_size)).zero_() #产生N个全零卷积核, Double类型的张量
        index = zeros([m+1,]*dim, dtype=np.int64)  # dim维，max_order+1阶

        for i, o in enumerate(self.flat_order_bank()):
            # print("in", "__init__")
            _torch_setter_by_index(moment[i], o, 1)
            _torch_setter_by_index(index, o, i)
            # moment[i,*o] = 1
            # index[*o] = i
        # Parameters are just Tensors limited to the module they are defined (in the module constructor method).__init__
        # All the filters involved in the PDE-Net are properly constrained using their associated moment matrices.
        # all filters are learned subjected to partial constraints on their associated moment matrices
        # self.kernel = nn.Parameter(kernel)
        self.moment = nn.Parameter(moment)
        self._index = index
        scale = torch.from_numpy(ones((self.moment.size()[0])))
        l = lambda a,b:a*b
        for i,o in enumerate(self.flat_order_bank()):
            # 貌似是多项式次数逐渐增加并累加, 总阶数小于最大order
            # 按照self._order_bank对应阶次生成kernel
            s = reduce(l, (self.dx[j]**oj for j,oj in enumerate(o)), 1) #从左到右两两进行lambda定义的操作，reduce到一个数
            scale[i] = 1/s
        self.register_buffer('scale',scale) # 在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出。

    def __index__(self,*args):
        return self.moment[_torch_reader_by_index(self._index, args)]

    def dim(self):
        return self._dim
    @property # 属性将是一个只读属性 getter
    def dx(self):
        return self._dx.copy()

    def kernel(self):
        scale = Variable(self.scale[:,newaxis])
        # print(scale)
        kernel = self.m2k(self.moment) # moment to kernel
        # print("... initialize finite difference kernels ...")
        # self.plt_kernels(kernel)
        size = kernel.size()
        kernel = kernel.view([size[0],-1]) # kernel dim0 是个数，保持不变，其余n*n部分展开为1*n**2
        # print((kernel*scale).view(size)[:,newaxis])
        return (kernel*scale).view(size)[:,newaxis] #scale the kernel, and 增加一个维度 (为了batch？)

    def plt_kernels(self, kernel):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=self._max_order + 1, ncols=self.max_num_kernel_each_order, sharex='col',
                               sharey='row')
        axes = ax.flatten()
        i = 0
        for order, order_list in enumerate(self._order_bank):
            for no_order_pair, order_pair in enumerate(order_list):
                i += 1
                axes[order * self.max_num_kernel_each_order + no_order_pair].matshow(kernel.detach().cpu().data[i - 1])
                axes[order * self.max_num_kernel_each_order + no_order_pair].set_title(
                    "$D{}{}$".format(order_pair[0], order_pair[1]))
        for i in axes:
            i.axis('off')
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()

    def flat_order_bank(self): # 在一个函数中，使用yield关键字，则当前的函数会变成生成器， yield可以从之前中断的地方继续执行
        for a in self._order_bank: # 取出每个 order bank
            for o in a: # 输出各个order分量
                yield o

    def _proj_(self,M,s,c):# imposing constraints on M
        for j in range(s): # 几阶导数算子
            for o in self._order_bank[j]: # 哪个方向的导数算子
                _torch_setter_by_index(M,o,c)
                # M[*o] = c
    # proj: 对训练迭代中施加限制
    def x_proj(self,*args,**kw):
        if self.constraint == 'free':
            return None
        if isinstance(self.constraint,int):  # acc = which type of constraint
            acc = self.constraint
        else:
            acc = 1
        for i,o in enumerate(self.flat_order_bank()):  #
            # order = sum(o)
            self._proj_(self.moment.data[i], sum(o)+acc, 0)  # 对moment的权值某些entry置零
            _torch_setter_by_index(self.moment.data[i], o, 1)   # 对moment某些entry置零
            # self.moment.data[i,*o] = 1
        return None
    def grad_proj(self,*args,**kw):
        if self.constraint == 'free':
            return None
        if isinstance(self.constraint,int): # acc = which type of constraint
            acc = self.constraint
        else:
            acc = 1
        for i,o in enumerate(self.flat_order_bank()):
            self._proj_(self.moment.grad.data[i],sum(0)+acc,0) # 对moment的权值梯度某些entry置零（不予更新）
        return None
    # MomentBank功能：产生 kernel
    def forward(self):
        return self.kernel()
#%%

class _FDNd(nn.Module):
    """
    Finite difference automatically handle boundary conditions
    Arguments for class:`_FDNd`:
        dim (int): dimension
        kernel_size (tuple of int): finite difference kernel size
        boundary (string): 'Dirichlet' or 'Periodic'
    Arguments for class:`MomentBank`:
        max_order, dx, constraint
    """
    def __init__(self, dim, kernel_size, boundary='Dirichlet'):
        super(_FDNd, self).__init__()
        self._dim = dim
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size,]*self._dim # [dim*dim*....*dim] 大小的 kernel
        self._kernel_size = kernel_size.copy()
        padwidth = [] #各个维度需要补零的厚度
        for k in reversed(kernel_size): # 反向
            padwidth.append((k-1)//2)
            padwidth.append(k-1-(k-1)//2)
        self._padwidth = padwidth
        self.boundary = boundary.upper() # 转为大写

    def dim(self):
        return self._dim
    @property
    def padwidth(self):
        return self._padwidth.copy()
    @property
    def boundary(self):
        return self._boundary
    @boundary.setter
    def boundary(self,v):
        self._boundary = v.upper()
    def pad(self, inputs): # 对输入数据补零
        if self.boundary == 'DIRICHLET':
            return F.pad(inputs, self.padwidth)  # 矩阵填充函数，input：四维或者五维的tensor Variabe， pad：不同Tensor的填充方式
        elif self.boundary == 'PERIODIC':
            return periodicpad(inputs, self.padwidth)

    # 卷积方式：inputs*weights
    def conv(self, inputs, weight):
        raise NotImplementedError
    # _FDNd功能：先补零后卷积
    def forward(self, inputs, kernel): # 实现用自己kernel的conv，pad zero + conv
        """
        Arguments:
            inputs (Variable): torch.size: (batch_size, spatial_size[0], spatial_size[1], ...)
        """

        inputs = self.pad(inputs)
        # inputs = inputs[:,newaxis]

        return self.conv(inputs, kernel) #kernel为自定义的

class FD1d(_FDNd):
    def __init__(self, kernel_size, max_order, dx=1.0, constraint='moment', boundary='Dirichlet'):
        super(FD1d, self).__init__(1, kernel_size, boundary=boundary)
        self.MomentBank = MomentBank(1, kernel_size, max_order, dx=dx, constraint=constraint)
        self.conv = F.conv1d
        self.kernel = self.MomentBank.kernel
class FD2d(_FDNd): # main network name
    def __init__(self, kernel_size, max_order, dx=1.0, constraint='moment', boundary='Dirichlet'):
        super(FD2d, self).__init__(2, kernel_size, boundary=boundary)
        self.MomentBank = MomentBank(2, kernel_size, max_order, dx=dx, constraint=constraint)
        self.conv = F.conv2d  # 在输入图像input中使用filters做卷积运算，实现_FDNd中的conv
        # self.kernel = self.MomentBank.kernel
class FD3d(_FDNd):
    def __init__(self, kernel_size, max_order, dx=1.0, constraint='moment', boundary='Dirichlet'):
        super(FD3d, self).__init__(3, kernel_size, boundary=boundary)
        self.MomentBank = MomentBank(3, kernel_size, max_order, dx=dx, constraint=constraint)
        self.conv = F.conv3d
        # self.kernel = self.MomentBank.kernel

