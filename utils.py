import torch
from functools import reduce

__all__ = ['tensordot','periodicpad', 'roll']

def periodicpad(inputs, pad):
    """
    'periodic' pad, similar to torch.nn.functional.pad
    周期填充边界
    """
    # print(pad)
    n = inputs.dim()
    inputs = inputs.permute(*list(range(n-1,-1,-1))) # 维度反转
    pad = iter(pad) # pad width
    i = 0  # i的大小是受限于维度dim，目的是对空间域进行周期padding
    indx = []
    for a in pad:
        b = next(pad)
        # print(a,b)
        assert a<inputs.size()[i] and b<inputs.size()[i]
        permute = list(range(n))
        permute[i] = 0
        permute[0] = i
        inputs = inputs.permute(*permute)  # 把第i个dim转换到第0个dim，便于后边操作
        inputlist = [inputs,]
        if a > 0:
            inputlist = [inputs[slice(-a,None)],inputs]  # 最后a个放到最前
        if b > 0:
            inputlist = inputlist+[inputs[slice(0,b)],]  # 前b个放到最后
        if a+b > 0:
            inputs = torch.cat(inputlist,dim=0)  # 合并
        inputs = inputs.permute(*permute)
        i += 1
    inputs = inputs.permute(*list(range(n-1,-1,-1)))  # 回归原有顺序
    return inputs



def tensordot(a,b,dim):
    """
    tensordot in PyTorch, see numpy.tensordot?
    """
    l = lambda x, y:x * y
    if isinstance(dim,int):
        a = a.contiguous()
        b = b.contiguous()
        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-dim]  # torch.Size 中 0, ..., total - dim 的位置上显示的size  d1 = total - dim +1
        sizea1 = sizea[-dim:]  # torch.Size 中 total - dim, ..., total  的位置上显示的size d2=dim+1
        sizeb0 = sizeb[:dim]  # torch.Size 中 0, ..., dim 的位置上显示的size d2=dim+1
        sizeb1 = sizeb[dim:] # torch.Size 中 dim, ..., total  的位置上显示的size d1 = total - dim +1
        N = reduce(l, sizea1, 1) # torch.Size 中 total - dim, ..., total 的位置上显示的size的连乘积
        assert reduce(l, sizeb0, 1) == N
    else:
        adims = dim[0]
        bdims = dim[1]
        adims = [adims,] if isinstance(adims, int) else adims # 是个list
        bdims = [bdims,] if isinstance(bdims, int) else bdims # 是个list
        # tensor的维度(0,1,...,dim) 与 dim[0] 的差集 aka, 小于dim of tensor 的那个维度不在dim[0]中
        adims_ = set(range(a.dim())).difference(set(adims))
        adims_ = list(adims_)
        adims_.sort()
        perma = adims_+adims # 把dim[0]补充在从小到大排列的小于dim of tensor a 的那些个维度的后面
        # difference() 方法用于返回集合的差集，即返回的集合元素包含在第一个集合中，但不包含在第二个集合(方法的参数)中
        bdims_ = set(range(b.dim())).difference(set(bdims))
        bdims_ = list(bdims_)
        bdims_.sort() # 是个list 从小到大排序
        permb = bdims+bdims_ # 把dim[1]补充在从小到大排列的小于dim of tensor b 的那些个维度的后面

        a = a.permute(*perma).contiguous()
        b = b.permute(*permb).contiguous()

        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-len(adims)]  #新添加的（dim[0]中的）维度个数 d1
        sizea1 = sizea[-len(adims):] #total - 新添加的（dim[1]中的）维度个数 d2
        sizeb0 = sizeb[:len(bdims)]  #total - 新添加的（dim[1]中的）维度个数 d2
        sizeb1 = sizeb[len(bdims):] #新添加的（dim[1]中的）维度个数 d1
        N = reduce(l, sizea1, 1) # 连乘积
        assert reduce(l, sizeb0, 1) == N
    # 不知道你想要多少行,但确定列数,那么你可以将行数设置为-1 只有一个轴值可以是-1)。这是告诉系统Library：给我一个具有这么多列的张量，并计算实现这一点所需的适当行数
    a = a.view([-1,N]) # ?*N
    b = b.view([N,-1]) # N*?
    c = a@b
    return c.view(sizea0+sizeb1) #2*[d1] = [d1, d1]

def _roll(inputs, shift, axis):
    shift = shift%inputs.shape[axis]
    if shift == 0:
        return inputs
    idx1 = [slice(None),]*inputs.dim()
    idx2 = [slice(None),]*inputs.dim()
    idx1[axis] = slice(None,-shift)
    idx2[axis] = slice(-shift,None)
    return torch.cat([inputs[idx2],inputs[idx1]], dim=axis)
def roll(inputs, shift, axis=None):
    """
    roll in PyTorch, see numpy.roll?
    """
    shape = inputs.shape
    if axis is None:
        if not isinstance(shift, int):
            shift = sum(shift)
        inputs = inputs.flatten()
        inputs = _roll(inputs, shift, axis=0)
        inputs = inputs.view(shape)
        return inputs
    if isinstance(axis, int):
        assert isinstance(shift, int)
        axis = [axis,]
        shift = [shift,]
    for (s,a) in zip(shift, axis):
        inputs = _roll(inputs, s, a)
    return inputs
