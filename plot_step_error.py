#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys,os
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
from scipy.optimize import fmin_bfgs as bfgs
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import NFI
import pdedata_fixsrc as pdedata
import nonlinpdeconfig_fixsrc as nonlinpdeconfig
import matplotlib.pyplot as plt
import parameters
import inspect
from gpu_mem_track import MemTracker  # 引用显存跟踪代码

device = torch.device('cuda:0')

frame = inspect.currentframe()
gpu_tracker = MemTracker(frame)      # 创建显存检测对象

gpu_tracker.track()

outfilename = "Figures/" + "NN_vs_FDM.gif"
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

options = parameters._options
options = nonlinpdeconfig.setoptions(argv=sys.argv[1:], kw=options, configfile=None)

#%% training
step_list = np.linspace(1550,3000, 30, dtype=int)
print(step_list)
filter_name = ['D01', 'D10', 'D20', 'D11', 'D02']
error_list_net = []

for step in step_list:
    namestobeupdate, _, fdmlearner = nonlinpdeconfig.setenv(options)
    def x_proj(*args, **kw):
        fdmlearner.fd2d_u1.MomentBank.x_proj()
        fdmlearner.id_u1.MomentBank.x_proj()
        fdmlearner.id_u0.MomentBank.x_proj()

    def grad_proj(*args, **kw):
        fdmlearner.fd2d_u1.MomentBank.grad_proj()
        fdmlearner.id_u1.MomentBank.grad_proj()
        fdmlearner.id_u0.MomentBank.grad_proj()

    globals().update(namestobeupdate)  # 其实是吧options记录到global
    d = pdedata.fdm2d(T=step*dt, mesh_size=(parameters._parameters.DIM, parameters._parameters.DIM)
                      , freq=parameters._parameters.FREQUENCY, boundary=boundary)
    dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size, num_workers=0)
    dataloader = iter(dataloader)
    error_net = 0
    for epoch in range(1):
        sample = pdedata.ToVariable()(pdedata.ToDevice(gpu)(pdedata.ToPrecision(precision)
                                                            (pdedata.AddNoise(0, 0)(next(dataloader)))))
        train_data = (sample['u1'][:, np.newaxis, :, :], sample['u0'][:, np.newaxis, :, :],
                      sample['epsr'][:, np.newaxis, :, :], sample['sigma'][:, np.newaxis, :, :])
        isfrozen = False
        forward = lambda: torch.norm((fdmlearner(train_data, step).squeeze() - sample['uT']))
        nfi = NFI.NumpyFunctionInterface(
            [dict(params=fdmlearner.diff_params(), isfrozen=isfrozen, x_proj=x_proj, grad_proj=grad_proj)],
            forward=forward, always_refresh=True)
        try:
            xopt = bfgs(nfi.f, nfi.flat_param, nfi.fprime, gtol=1e-15, maxiter=500, disp=False)
            #xopt = np.array([-9.3703367e-10,  2.6087457e-09,  1.0000000e+00,  3.0167921e-08, 1.0000001e+00])
        except RuntimeError as Argument:
            pass
        finally:
            # save parameters
            nfi.flat_param = xopt
            loss = forward().item()
            error_net = loss / torch.norm(sample['uT']).item()
            print('epoch {}, step {}, error net {}.'.format(epoch, step, error_net))

        gpu_tracker.track()
        del loss
        del train_data
        del nfi
        del sample
        torch.cuda.empty_cache()
    error_list_net.append(error_net)
    del fdmlearner
    del namestobeupdate
    torch.cuda.empty_cache()


plt.figure()
plt.plot(step_list, error_list_net, 'k-*', label='NN error')
plt.legend()
plt.xlabel('step number')
plt.ylabel('error')
plt.title('noise intensity - relative error')
plt.savefig("Figures/" + 'step length - relative error')
np.save('1550-3000.npy', error_list_net)
plt.show()