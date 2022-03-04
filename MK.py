"""Moment(sum rules) and Kernel(convolution kernel) convertor"""
from numpy import *
from numpy.linalg import *
from scipy.special import factorial
from functools import reduce
import torch
import torch.nn as nn
from utils import tensordot

__all__ = ['M2K','K2M']

def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim()-1
    for i in range(k):
        x = tensordot(mats[k-i-1], x, dim=[1,k])
    x = x.permute([k,]+list(range(k))).contiguous() #最后一列换到第一列
    x = x.view(sizex) #保持原来形状
    return x

def _apply_axis_right_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim() - 1
    x = x.permute(list(range(1, k+1))+[0,]) # 第一列换为最后一列
    for i in range(k):
        x = tensordot(x, mats[i], dim=[0,0])
    x = x.contiguous()
    # 将输入的torch.Tensor改变形状(size)并返回.返回的Tensor与输入的Tensor必须有相同的元素,
    x = x.view(sizex)  # 保持原来形状， 相同的元素数目, 但形状可以不一样 即, view起到的作用是reshape, view的参数的是改变后的shape.
    return x

class _MK(nn.Module):
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
            invM.append(inv(M[-1])) #最新正方形（l*l）的tensor的逆tensor
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

    def forward(self):
        pass

class M2K(_MK): # All the filters involved in the PDE-Net are properly constrained using their associated moment matrices.
    """
    convert moment matrix to convolution kernel
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        m2k = M2K([5,5])
        m = torch.randn(5,5,dtype=torch.float64)
        k = m2k(m)
    """
    def __init__(self, shape):
        super(M2K, self).__init__(shape) # shape is kernal size
    def forward(self, m): #对输入的tensor进行操作
        """
        m (Tensor): torch.size=[...,*self.shape]
        """
        sizem = m.size()
        # print(m)
        m = self._packdim(m)
        # print(m)
        m = _apply_axis_left_dot(m, self.invM) # m*invM
        # print(m)
        m = m.view(sizem) #resize到原来大小
        return m
# m2k = M2K([2,3,3])
# m = torch.randn(2,3,3,dtype=torch.float64)
# k = m2k(m)
class K2M(_MK):
    """
    convert convolution kernel to moment matrix
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        k2m = K2M([5,5])
        k = torch.randn(5,5,dtype=torch.float64)
        m = k2m(k)
    """
    def __init__(self, shape):
        super(K2M, self).__init__(shape)
    def forward(self, k):
        """
        k (Tensor): torch.size=[...,*self.shape]
        """
        # print(k.size())
        sizek = k.size()
        k = self._packdim(k)
        # print(k.size())
        k = _apply_axis_left_dot(k, self.M)
        # print(k.size())
        k = k.view(sizek)
        # print(k.size())
        return k

