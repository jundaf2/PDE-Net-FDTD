#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys,os
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
import numpy as np
import torch
import torchvision
import NFI
import pdedata
import nonlinpdeconfig
import matplotlib.pyplot as plt
#
def plt_training_data(sample):
    print(sample['u0'].cpu().shape)
    print(sample['uT0'].cpu().shape)
    plt.figure()
    plt.imshow(sample['u0'][0].cpu())
    plt.figure()
    plt.imshow(sample['u1'][0].cpu())
    plt.figure()
    plt.imshow(sample['uT0'][0].cpu())
    plt.figure()
    plt.imshow(sample['uT1'][0].cpu())
    # plt.show()
def plt_output_data(learner, sample, step):
    print(learner(sample['u0'], step).detach().numpy().shape)
    print(learner(sample['u1'], step).detach().numpy().shape)
    plt.figure()
    plt.imshow(learner(sample['u0'], step).detach().squeeze().numpy())
    plt.figure()
    plt.imshow(learner(sample['u1'], step).detach().squeeze().numpy())
    plt.show()

options = { # defalt options
        '--precision':'double',
        '--taskdescriptor':'fdm2d-test',
        '--constraint':'frozen',
        '--gpu': -1, #0,
        '--kernel_size':7,'--max_order':4,
        '--xn':'50','--yn':'50',
        '--interp_degree':2,'--interp_mesh_size':5,
        '--nonlinear_interp_degree':4, '--nonlinear_interp_mesh_size':20,
        '--nonlinear_interp_mesh_bound':15,
        '--nonlinear_coefficient':15,
        '--batch_size':1,'--teststepnum':10,
        '--maxiter':2500,
        '--dt':1e-6/40,
        '--dx':15.0,
        '--start_noise_level':0.01,'--end_noise_level':0.01,
        '--layer':list(range(0, 40)),
        '--recordfile':'convergence',
        '--recordcycle':100, '--savecycle':10000,
        '--repeatnum':5,
        }
options = nonlinpdeconfig.setoptions(argv=sys.argv[1:], kw=options, configfile=None)

namestobeupdate, callback, fdmlearner = nonlinpdeconfig.setenv(options)

#globals() method returns the dictionary of the current global symbol table. A symbol table is a data structure maintained
# by a compiler which contains all necessary information about the program. These include variable names, methods, classes
globals().update(namestobeupdate) # 其实是吧options记录到global

#%% training
trans = None #torchvision.transforms.Compose([pdedata.DownSample(5,'Dirichlet'), pdedata.ToTensor(), pdedata.ToPrecision(precision), pdedata.AddNoise(start_noise_level, end_noise_level)])
for l in layer:
    print('layer{}:'.format(l))
    # 输出该层信息到文件
    if l == 0:
        callback.stage = 'warmup'
        isfrozen = True
    else:
        callback.stage = 'layer-'+str(l)
        if constraint == 'moment' or constraint == 'free':
            isfrozen = False
        elif constraint == 'frozen':
            isfrozen = True
    step = (l if l>=1 else 1)
    # generate layer-l data
    d = pdedata.fdm2d(T=step*dt, mesh_size=(20, 20), freq=1e6)
    # pytorch采用 for x in iterator 模式，从Dataloader类中读取数据
    dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size, num_workers=0)
    dataloader = iter(dataloader) # next函数，输入迭代器，调用__next__，取出数据
    sample = pdedata.ToVariable()(pdedata.ToDevice(gpu)(next(dataloader))) # next函数，输入迭代器，调用__next__，取出数据
    # size of tensor sample['uT0'] --> [batch size]+mesh_size
    del dataloader
    xy = torch.stack([sample['x'],sample['y']], dim=3)
    fdmlearner.xy = xy # set xy for pde-net
    # set NumpyFunctionInterface
    # plt_training_data(sample)
    # plt_output_data(fdmlearner, sample, step)
    # train_data = torch.cat((sample['u1'], sample['u0'], sample['j2'], sample['j0']), dim=0)
    forward = lambda:torch.mean((fdmlearner(sample['u0'], step)-sample['uT'])**2)
    def x_proj(*args,**kw):
        fdmlearner.id.MomentBank.x_proj()
        fdmlearner.fd2d.MomentBank.x_proj()
    def grad_proj(*args,**kw):
        fdmlearner.id.MomentBank.grad_proj()
        fdmlearner.fd2d.MomentBank.grad_proj()
    # nfi = NFI.NumpyFunctionInterface(
    #         [dict(params=fdmlearner.diff_params(), isfrozen=isfrozen, x_proj=x_proj, grad_proj=grad_proj),
    #             dict(params=fdmlearner.coe_params(), isfrozen=False, x_proj=None, grad_proj=None)],
    #         forward=forward, always_refresh=False)
    nfi = NFI.NumpyFunctionInterface(
        [dict(params=fdmlearner.diff_params(), isfrozen=isfrozen, x_proj=x_proj, grad_proj=grad_proj)],
        forward=forward, always_refresh=False)
    callback.nfi = nfi
    try:
        # 最小化函数（fmin_l_bfgs_b）允许传回函数值f（x）及其渐变f'（x），在前面的步骤中计算过
        xopt,f,d = lbfgsb(nfi.f, nfi.flat_param, nfi.fprime, m=500, callback=callback,
                          factr=1e3, pgtol=1e-7, maxiter=maxiter, iprint=50)
    except RuntimeError as Argument:
        with callback.open() as output:
            print(Argument, file=output) # if overflow then just print and continue
    finally: # 无论是否发生异常，只要提供了finally程序，就在执行所有步骤之后执行finally中的程序
        # save parameters
        print('... save parameters ...')
        nfi.flat_param = xopt
        callback.save(xopt, 'final')