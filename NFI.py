"""numpy function interface for torch"""
import torch
from functools import reduce
import warnings
import numpy as np

import PGManager

__all__ = ['NumpyFunctionInterface',]

class NumpyFunctionInterface(PGManager.ParamGroupsManager):
    """
    Interfaces class for representing torch forward & backward procedures 
           as Numpy functions. 
           `NumpyFunctionInterface` contains two part of compositions: 
           which are parameters group manager 
           (see docstrings of `ParamGroupsManager`)
           and numpy function interface 
           (pls refer to aTEAM/test/optim_quickstart.py).

    .. warning::
    If self.always_refresh=False, and you are going to change options of one 
        of the param_groups, please use self.set_options. This is because, for 
        example, any changes on 'grad_proj's will have impact on self.fprime(x), 
        even for the same input x; so do any changes on 'isfrozen's, 'x_proj's. 

    .. warning::
    Right now all parameters have to be dense Variable and their dtype 
        (float or double) have to be the same. This will be improved in the 
        future.

    Arguments:
        params (iterable): See ParamGroupsManager.__doc__
        forward (callable): callable forward(**kw)
            torch forward procedure, return a :class:`torch.Tensor`
        always_refresh (bool): If always_refresh=True, then any changes on 
            forward & backward procedure is OK. We recommand you to set 
            always_refresh=True unless you are familiar with 
            :class:`NumpyFunctionInterface`.
            When always_refresh=False, NumpyFunctionInterface will cache 
            parameters for fast forward & backward.
        # >>>> set default options for parameter groups >>>>
            # Each parameter group will have its own options dict,
            # (see ParamGroupsManager.__doc__)
            # A special option for idx-th param_group can be set up by calling 
            #       self.set_options(idx, optionsdict)
            # e.g., self.set_options(0, {'x_proj':x_proj})
        isfrozen (bool): whether parameters should be frozen, if you set 
            isfrozen=True, as a result, grad of this param_group would be 
            set to be 0 after calling self.fprime(x).
        x_proj (callable): callable x_proj(param_group['params']). 
            It is similar to nn.module.register_forward_pre_hook(x_proj)
            It can be used to make parameters to satisfied linear constraint. 
            Wether isfrozen or not, x_proj&grad_proj will go their own way.
        grad_proj (callable): callable grad_proj(param_group['params']).
            It is similar to nn.module.register_backward_hook(grad_proj).
            grad_proj(param_group['params']) should project gradients of 
            param_group['params'] to the constrained linear space if needed.
        **kw (keyword args): other options for parameter groups
        <<<< set default options for parameter groups <<<<
    """
    # 可变参数允许你传入0个或任意个参数，这些可变参数在函数调用时自动组装为一个tuple,
    # 而关键字参数允许你传入0个或任意个含参数名的参数，这些关键字参数在函数内部自动组装为一个dict
    # ** extra表示把extra这个dict的所有key - value用关键字参数传入到函数的 ** kw参数，
    # kw将获得一个dict，注意kw获得的dict是extra的一份拷贝，对kw的改动不会影响到函数外的extra
    def __init__(self, params, forward, always_refresh=True, *, 
            isfrozen=False, x_proj=None, grad_proj=None, **kw):
        defaults = dict(isfrozen=isfrozen, x_proj=x_proj, grad_proj=grad_proj, **kw)
        super(NumpyFunctionInterface, self).__init__(params, defaults)
        # print(params)
        self.dtype = next(self.params).data.cpu().numpy().dtype
        self._forward = forward
        self.options_refresh()
        self.always_refresh = always_refresh
        # print("... parameter group number is {} ...".format(len(params[0]['params'])))

    def options_refresh(self):
        """
        Any changes on 'isfrozen's, 'x_proj's, 'grad_proj's, self._forward will 
        have impact on self.f, self.fprime. Call this function to keep them 
        safe when you apply any changes on options.
        """
        self._need_backward = True
        self._grad_cache = None
        self._x_cache = None
        self._loss = None
        self._numel = None

    @staticmethod
    def _proj_check(kw):
        if not kw['isfrozen'] and None in set([kw['x_proj'],kw['grad_proj']]):
            if not (kw['x_proj'] is None and kw['grad_proj'] is None):
                warnings.warn("Exactly one of {x_proj,grad_proj} is not None, "
                        "and the parameters are not set to be frozen, "
                        "make sure what you are doing now.")
        return None
    def set_options(self, idx, **kw):
        """
        A safe way to update idx_th param_group's options.
        """
        self.param_groups[idx].update(**kw)
        NumpyFunctionInterface._proj_check(self.param_groups[idx])
        self.options_refresh()
    def add_param_group(self, param_group):
        super(NumpyFunctionInterface, self).add_param_group(param_group)
        param_group_tmp = self.param_groups[-1]
        # check consistency of x_proj, grad_proj, isfrozen
        NumpyFunctionInterface._proj_check(param_group_tmp)
        # check is_leaf, requires_grad
        for pgn,p in param_group_tmp['params'].items():
            # print("... parameter group {} ...".format(pgn))
            # tensor可细分为两类：叶子节点(leaf node)和非叶子节点
            if not p.is_leaf:
                raise ValueError("can't manage a non-leaf Tensor")
            if not p.requires_grad:
                raise ValueError("managing a Tensor that does not "
                        "require gradients")
        self.options_refresh()

    @property
    def forward(self):
        """
        A safe way to get access of self._forward.
        When you use property NumpyFunctionInterface.forward, after 
        doing some modifications on self._forward, like: 
            self.forward.property = value
        in this case, we should call self.options_refresh() to keep self.f and 
        self.fprime safe. 
        """
        self.options_refresh()
        return self._forward
    @forward.setter
    def forward(self, v):
        self.options_refresh()
        self._forward = v

    def numel(self):
        if not self._numel is None:
            return self._numel
        return reduce(lambda a,p: a+p.numel(), self.params, 0)

    def _all_x_proj(self):
        for param_group in self.param_groups:
            x_proj = param_group['x_proj']
            if not x_proj is None:
                x_proj(param_group['params'])
    def _all_grad_proj(self):
        for param_group in self.param_groups:
            grad_proj = param_group['grad_proj']
            if not grad_proj is None:
                grad_proj(param_group['params'])

    # if you do self.flat_param = x; y = self.flat_param; 
    # np.array_equal(x,y) may not be True.
    # because of 'x_proj's,'grad_proj's may have impact on self.flat_param
    @property
    def flat_param(self):
        views = []
        self._all_x_proj()
        for p in self.params:
            view = p.data.view(-1).cpu()
            views.append(view)
        return torch.cat(views,0).numpy()
    @flat_param.setter
    def flat_param(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.numel()
        x = x.astype(dtype=self.dtype,copy=False)
        offset = 0
        for isfrozen,p in self.params_with_info('isfrozen'):
            numel = p.numel() # 获取tensor中一共包含多少个元素
            if not isfrozen:
                p_tmp = torch.from_numpy(x[offset:offset+numel]).view_as(p)
                p.data.copy_(p_tmp)
            offset += numel
        self._all_x_proj()

    def _flat_grad(self):
        views = []
        self._all_grad_proj()
        for isfrozen, p in self.params_with_info('isfrozen'):
            if isfrozen or p.grad is None:
                view = torch.zeros(p.numel(), dtype=p.dtype)
            else:
                view = p.grad.data.view(-1).cpu()
            views.append(view)
        return torch.cat(views, 0).numpy()

    def f(self, x, *args, **kw): # function evaluation, 等待优化函数f是损失函数loss (residual)
        """
        self.f(x) depends on self.flat_param and self.forward
        """
        if self.always_refresh:
            self.options_refresh()
        self.flat_param = x
        _x_cache = self.flat_param
        if self._loss is None or not np.array_equal(_x_cache, self._x_cache):
            self._x_cache = _x_cache
            self._loss = self._forward() # forward 就是计算loss的lambda表达式
            if torch.isnan(self._loss):
                self._loss = (torch.ones(1,
                    requires_grad=self._loss.requires_grad)/
                    torch.zeros(1))#.to(loss)

            self._need_backward = True
        return self._loss.item()
    def fprime(self, x, always_double=True, *args, **kw): # function derivative (BP)
        self.f(x) # 计算完loss才有backward
        if self._need_backward:
            self.zero_grad()
            self._loss.backward() # 调用 tensor 的 自动微分（AD）
            self._grad_cache = self._flat_grad()
        self._need_backward = False  # backward完毕，取消need
        if always_double:
            return self._grad_cache.astype(np.float64)
        else:
            return self._grad_cache


