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
import pdedata
import nonlinpdeconfig
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
step = parameters._parameters.STEP
filter_name = ['D01', 'D10', 'D20', 'D11', 'D02']
error_list_net = []
error_list_fdm = []
intensity_list = list(np.linspace(0, 0.3, 10))

for intensity in intensity_list:
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

    error_net = 0
    error_fdm = 0
    count = 0
    for epoch in range(5):
        dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size, num_workers=0)
        dataloader = iter(dataloader)
        sample = pdedata.ToVariable()(pdedata.ToDevice(gpu)(pdedata.ToPrecision(precision)
                                                            (pdedata.AddNoise(intensity, intensity)(next(dataloader)))))
        train_data = (sample['u1'][:, np.newaxis, :, :], sample['u0'][:, np.newaxis, :, :]
                      , sample['j2set'][:, :, :, :], sample['j0set'][:, :, :, :],
                      sample['epsr'][:, np.newaxis,:,:], sample['sigma'][:, np.newaxis,:,:]
                      )
        isfrozen = False
        forward = lambda: torch.norm((fdmlearner(train_data, step).squeeze() - sample['uT']))
        nfi = NFI.NumpyFunctionInterface(
            [dict(params=fdmlearner.diff_params(), isfrozen=isfrozen, x_proj=x_proj, grad_proj=grad_proj)],
            forward=forward, always_refresh=True)
        try:
            xopt = bfgs(nfi.f, nfi.flat_param, nfi.fprime, gtol=1e-12, maxiter=500, disp=False)

        except RuntimeError as Argument:
            pass
        finally:
            # save parameters
            print(xopt)
            nfi.flat_param = xopt
            loss = forward().item()
            _error_net = loss / torch.norm(sample['uT_clean']).item()
            _error_fdm = torch.norm((sample['uT_clean'] - sample['uT'])).item() / torch.norm(sample['uT_clean']).item()
            if _error_net<0.4:
                error_net += _error_net
                error_fdm += _error_fdm
                count += 1
            print('epoch {}, error net {}, error fdm {}.'.format(epoch, _error_net, _error_fdm))
        gpu_tracker.track()
        del loss
        del train_data
        del nfi
        del sample
        del dataloader
        torch.cuda.empty_cache()
    error_list_net.append(error_net/count)
    error_list_fdm.append(error_fdm/count)
    del fdmlearner
    del namestobeupdate


plt.figure()
plt.plot(intensity_list, error_list_net, 'k--o', label='NN error')
plt.plot(intensity_list, error_list_fdm, 'k-*', label='FDM error')
plt.legend()
plt.xlabel('I')
plt.ylabel('error')
plt.title('noise intensity - relative error')
plt.savefig("Figures/" + 'noise intensity - relative error')
plt.show()