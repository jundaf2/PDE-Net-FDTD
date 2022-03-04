#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import os,sys,contextlib
import numpy as np
import torch
from torch.autograd import Variable
import getopt,yaml,time
import combo_fixsrc_learner as combo_learner
#%%
def _options_cast(options, typeset, thistype):
    for x in typeset:
        options['--'+x] = thistype(options['--'+x])
        # print(x, options['--'+x])
    return options
def _option_analytic(option, thistype):
    if not isinstance(option, str):
        return option
    l0 = option.split(',')
    l = []
    for l1 in l0:
        try:
            ll = thistype(l1)
            x = [ll,]
        except ValueError:
            z = l1.split('-')
            x = list(range(int(z[0]), int(z[1])+1))
        finally:
            l = l+x
    return l
def _setoptions(options):
    assert options['--precision'] in ['float','double']
    # str options
    strtype = ['taskdescriptor', 'constraint', 'recordfile']
    options = _options_cast(options, strtype, str)
    assert options['--constraint'] in ['frozen','moment','free']
    # int options
    inttype = ['gpu', 'kernel_size', 'max_order', 'xn', 'yn', 'interp_degree', 'interp_mesh_size', 'nonlinear_interp_degree', 'nonlinear_interp_mesh_size', 
             'batch_size', 'teststepnum', 'maxiter', 'recordcycle', 'savecycle', 'repeatnum']
    options = _options_cast(options, inttype, int)
    # float options
    floattype = ['dt', 'start_noise_level', 'end_noise_level', 'nonlinear_interp_mesh_bound']
    options = _options_cast(options, floattype, float)

    options['--layer'] = list(_option_analytic(options['--layer'], int))
    return options
def setoptions(*, argv=None, kw=None, configfile=None, isload=False):
    """
    proirity: argv>kw>configfile
    Arguments:
        argv (list): command line options
        kw (dict): options
        configfile (str): configfile path
        isload (bool): load or set new options
    """
    options = {
            '--precision':'double',
            '--taskdescriptor':'fdm2d',  #  linearity of Maxwell's equations
            '--constraint':'moment',
            '--gpu':-1,
            '--kernel_size':5,'--max_order':1,
            '--xn':'50','--yn':'50',
            '--interp_degree':2,'--interp_mesh_size':5,
            '--nonlinear_interp_degree':2, '--nonlinear_interp_mesh_size':20,
            '--nonlinear_interp_mesh_bound':15,
            '--nonlinear_coefficient':15,
            '--batch_size':2,'--teststepnum':10,
            '--maxiter':2000,
            '--dt':2.5e-8,
            '--dx':15.0,
            '--start_noise_level':0.01,'--end_noise_level':0.01,
            '--layer':list(range(0,5)),
            '--recordfile':'convergence',
            '--recordcycle':20,'--savecycle':1000,
            '--repeatnum':5,
            '--boundary':'PERIODIC',
            }
    longopts = list(k[2:]+'=' for k in options)
    longopts.append('configfile=')
    if not argv is None:
        options.update(dict(getopt.getopt(argv, shortopts='f',longopts=longopts)[0]))
    if '--configfile' in options:
        assert configfile is None, 'duplicate configfile in argv.'
        configfile = options['--configfile']
    if not configfile is None:
        options['--configfile'] = configfile
        with open(configfile, 'r') as f:
            options.update(yaml.load(f), Loader=yaml.SafeLoader)
    if not kw is None:
        options.update(kw) # 更新参数字典（kw or option）
    if not argv is None:
        options.update(dict(getopt.getopt(argv, shortopts='f',longopts=longopts)[0]))
    options = _setoptions(options)
    options.pop('-f',1)
    savepath = 'checkpoint/'+options['--taskdescriptor']
    if not isload:
        try:
            os.makedirs(savepath)
        except FileExistsError:
            os.rename(savepath, savepath+'-'+str(np.random.randint(2**16)))
            os.makedirs(savepath)
        with open(savepath+'/options.yaml', 'w') as f:
            print(yaml.dump(options), file=f)
    return options

class callbackgen(object): # 看似是记录网络训练每一次迭代时的参数
    def __init__(self, options, nfi=None, module=None, stage=None):
        self.taskdescriptor = options['--taskdescriptor']
        self.recordfile = options['--recordfile']
        self.recordcycle = options['--recordcycle']
        self.savecycle = options['--savecycle']
        self.savepath = 'checkpoint/'+self.taskdescriptor
        self.startt = time.time()
        self.Fs = []
        self.Gs = []
        self.ITERNUM = 0

    @property
    def stage(self):
        return self._stage
    @stage.setter
    def stage(self, v):
        self._stage = v
        with self.open() as output: # 输出信息到文件
            print('\n', file=output)
            print('current stage is: '+v, file=output)
    @contextlib.contextmanager
    def open(self): #打开 recordfile
        isfile = (not self.recordfile is None)
        if isfile:
            output = open(self.savepath+'/'+self.recordfile, 'a')
        else:
            # 原始的 sys.stdout 指向控制台 如果把文件的对象的引用赋给 sys.stdout，那么 print 调用的就是文件对象的 write 方法
            # 如果你还想在控制台打印一些东西的话，最好先将原始的控制台对象引用保存下来，向文件中打印之后再恢复 sys.stdout
            output = sys.stdout
        try:
            yield output
        finally:
            if isfile:
                output.close()
    
    # remember to set self.nfi,self.module,self.stage
    def save(self, xopt, iternum): # 保存 module.state_dict()
        self.nfi.flat_params = xopt
        try:
            os.mkdir(self.savepath+'/params')
        except:
            pass
        filename = self.savepath+'/params/'+str(self.stage)+'-xopt-'+str(iternum)
        torch.save(self.module.state_dict(), filename)
        return None
    def load(self, l):
        # 确定当前处于那个stage
        if l == 0:
            stage = 'warmup'
        else:
            stage = 'layer-'+str(l)
        filename = self.savepath+'/params/'+str(stage)+'-xopt-final'
        params = torch.load(filename)  # 加载参数
        # print("... parameter names:\n {} ...".format(self.module))
        return None
    def record(self, xopt, iternum, **args):
        self.Fs.append(self.nfi.f(xopt))  # elaluvate function
        self.Gs.append(np.linalg.norm(self.nfi.fprime(xopt)))  # elaluvate derivative
        stopt = time.time()
        with self.open() as output:
            print('iter: {:6d}'.format(iternum), '   time: {:.2f}'.format(stopt-self.startt), file=output)
            print('Func: {:.2e}'.format(self.Fs[-1]), ' |g|: {:.2e}'.format(self.Gs[-1]), file=output)
        self.startt = stopt
        return None
    # 方法的作用其实是把一个类的实例化对象变成了可调用对象，也就是说把一个类的实例化对象变成了可调用对象，只要类里实现了__call__（）方法就行
    # 相当于重载了  ()
    def __call__(self, xopt, **args): # callback的函数调用括号内就是 xopt， 即参数向量
        # record用于回溯，save用于以后加载
        # if self.ITERNUM%self.recordcycle == 0:
        #     self.record(xopt, iternum=self.ITERNUM, **args)
        # if self.ITERNUM%self.savecycle == 0:
        #     self.save(xopt, iternum=self.ITERNUM)
        self.ITERNUM += 1
        return None
#%%

def setenv(options):

    namestobeupdate = {}
    namestobeupdate['precision'] = options['--precision']
    namestobeupdate['taskdescriptor'] = options['--taskdescriptor']
    namestobeupdate['constraint'] = options['--constraint']
    namestobeupdate['gpu'] = options['--gpu']
    namestobeupdate['kernel_size'] = [options['--kernel_size'],]*2
    namestobeupdate['max_order'] = options['--max_order']
    namestobeupdate['mesh_size'] = np.array([options['--xn'],options['--yn']])
    namestobeupdate['interp_degree'] = options['--interp_degree']
    namestobeupdate['interp_mesh_size'] = [options['--interp_mesh_size'],]*2
    namestobeupdate['nonlinear_interp_degree'] = options['--nonlinear_interp_degree']
    namestobeupdate['nonlinear_interp_mesh_size'] = options['--nonlinear_interp_mesh_size']
    namestobeupdate['nonlinear_interp_mesh_bound'] = [-options['--nonlinear_interp_mesh_bound'],options['--nonlinear_interp_mesh_bound']]
    namestobeupdate['nonlinear_coefficient'] = options['--nonlinear_coefficient']
    namestobeupdate['batch_size'] = options['--batch_size']
    namestobeupdate['teststepnum'] = options['--teststepnum']
    namestobeupdate['maxiter'] = options['--maxiter']
    namestobeupdate['dt'] = options['--dt']
    namestobeupdate['dx'] = options['--dx']
    namestobeupdate['start_noise_level'] = options['--start_noise_level']
    namestobeupdate['end_noise_level'] = options['--end_noise_level']
    namestobeupdate['layer'] = options['--layer']
    namestobeupdate['recordfile'] = options['--recordfile']
    namestobeupdate['recordcycle'] = options['--recordcycle']
    namestobeupdate['savecycle'] = options['--savecycle']
    namestobeupdate['repeatnum'] = options['--repeatnum']
    namestobeupdate['boundary'] = options['--boundary']
    fdmlearner = combo_learner.SingleNonLinear2d(kernel_size=namestobeupdate['kernel_size'],
                                                    max_order=namestobeupdate['max_order'],
                                                    dx=namestobeupdate['dx'],
                                                    dt=namestobeupdate['dt'],
                                                    constraint=namestobeupdate['constraint'],
                                                    boudary=namestobeupdate['boundary']) # build pde-net
    if namestobeupdate['precision'] == 'double':
        fdmlearner.double()
    else:
        fdmlearner.float()
    if namestobeupdate['gpu'] >= 0:
        fdmlearner.cuda(namestobeupdate['gpu'])
    else:
        fdmlearner.cpu()
    callback = callbackgen(options) # some useful interface
    callback.module = fdmlearner
    return namestobeupdate, callback, fdmlearner