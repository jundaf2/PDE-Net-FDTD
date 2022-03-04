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

namestobeupdate, callback, fdmlearner = nonlinpdeconfig.setenv(options)

def x_proj(*args, **kw):
    fdmlearner.fd2d_u1.MomentBank.x_proj()
    fdmlearner.id_u1.MomentBank.x_proj()
    fdmlearner.id_u0.MomentBank.x_proj()


def grad_proj(*args, **kw):
    fdmlearner.fd2d_u1.MomentBank.grad_proj()
    fdmlearner.id_u1.MomentBank.grad_proj()
    fdmlearner.id_u0.MomentBank.grad_proj()
globals().update(namestobeupdate) # 其实是吧options记录到global

#%% training
step = parameters._parameters.STEP
h1 = plt.figure(1)
ax1 = h1.add_subplot(121)
ax2 = h1.add_subplot(122)
h2 = plt.figure(2)
ax = h2.add_subplot(111)


filter_name = ['D01', 'D10', 'D20', 'D11', 'D02']

for epoch in range(100):
    gif_images = []
    isfrozen = False
    callback.stage = 'layer-' + str(step)
    # pytorch采用 for x in iterator 模式，从Dataloader类中读取数据
    d = pdedata.fdm2d(T=step * dt, mesh_size=(parameters._parameters.DIM, parameters._parameters.DIM)
                      , freq=parameters._parameters.FREQUENCY, boundary=boundary)
    dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size, num_workers=0)
    dataloader = iter(dataloader)  # next函数，输入迭代器，调用__next__，取出数据
    sample = pdedata.ToVariable()(pdedata.ToDevice(gpu)(pdedata.ToPrecision(precision)(pdedata.AddNoise(0.0,0.0)(next(dataloader)))))
    # sample = pdedata.ToVariable()(pdedata.ToDevice(gpu)(pdedata.ToPrecision(precision)(next(dataloader))))

    del dataloader

    train_data = (sample['u1'][:, np.newaxis, :, :], sample['u0'][:, np.newaxis, :, :]
                  , sample['j2set'][:, :, :, :], sample['j0set'][:, :, :, :],
                  sample['epsr'][:, np.newaxis,:,:], sample['sigma'][:, np.newaxis,:,:]
                  )

    forward = lambda: torch.norm((fdmlearner(train_data, step).squeeze() - sample['uT']))



    nfi = NFI.NumpyFunctionInterface(
        [dict(params=fdmlearner.diff_params(), isfrozen=isfrozen, x_proj=x_proj, grad_proj=grad_proj)],
        forward=forward, always_refresh=True)

    callback.nfi = nfi

    try:
        # drawing bar figure of parameters
        ax.clear()
        # ax.bar(x=np.arange(5), height=np.array([0, 0, 1.0, 0, 1.0]), bottom=0, label='Truth')
        kernel_number = np.sum(np.arange(fdmlearner.fd2d_u1.MomentBank.max_num_kernel_each_order)+1)-1
        ax.bar(x=np.arange(kernel_number), height=nfi.flat_param[-kernel_number:], bottom=0, label='Estimation')
        ax.legend()
        # ax.set_xticks(range(len(filter_name)))
        # ax.set_xticklabels(filter_name)
        # h2.savefig("Figures/" + 'weight epoch {}'.format(epoch))
        plt.pause(1)

        # 最小化函数（fmin_l_bfgs_b）允许传回函数值f（x）及其渐变f'（x），在前面的步骤中计算过
        xopt = bfgs(nfi.f, nfi.flat_param, nfi.fprime, gtol=1e-5, maxiter=500, disp=True, callback=callback)
        # xopt, f, b = lbfgsb(nfi.f, nfi.flat_param, nfi.fprime, m=500, callback=callback, factr=1e0, pgtol=1e-7,
        #                 maxiter=2000, iprint=50)
    except RuntimeError as Argument:
        with callback.open() as output:
            print(Argument, file=output) # if overflow then just print and continue
    finally: # 无论是否发生异常，只要提供了finally程序，就在执行所有步骤之后执行finally中的程序
        # save parameters
        nfi.flat_param = xopt
        # nfi.flat_param = np.array([0, 0, 1.0, 0, 1.0])
        # comparison with ground truth
        ax1.clear()
        ax2.clear()
        print(fdmlearner(train_data, step).size())
        print(sample['uT'].size())
        ufinial = fdmlearner(train_data, step).detach().squeeze().cpu().data
        ax1.imshow(ufinial[0], cmap='jet')
        ax2.imshow(sample['uT'][0].detach().squeeze().cpu().data, cmap='jet')
        ax1.set_title('NN Ouput')
        ax2.set_title('Ground Truth')
        h1.savefig("Figures/" + 'layer {} output {}'.format(step, epoch), bbox_inches='tight')
        plt.pause(1)

        # show total E-field energy of the simulation domain
        print('pde-cnn domain energy (norm): ', np.linalg.norm(ufinial[0]))
        print('fdm domain energy (norm): ', np.linalg.norm(sample['uT'][0].detach().squeeze().cpu().data))

        # print weights of the network kernels
        # print(fdmlearner.combofd.weight)
        # print(fdmlearner.id_u1.MomentBank.kernel())
        # print(fdmlearner.id_u0.MomentBank.kernel())
        print(fdmlearner.combofd.weight.permute(1,0,2,3).size(),fdmlearner.fd2d_u1.MomentBank.kernel().size(),(torch.sum(fdmlearner.combofd.weight*fdmlearner.fd2d_u1.MomentBank.kernel()[1:], dim=0)).size())
        print(torch.sum(fdmlearner.combofd.weight.permute(1,0,2,3)*fdmlearner.fd2d_u1.MomentBank.kernel()[1:], dim=0))

        # draw gif figure of transient em wave
        # for t in range(1, step):
        #     ax1.clear()
        #     ax1.imshow(fdmlearner(train_data, t)[0].detach().squeeze().cpu().data, cmap='jet')
        #     h1.canvas.draw()  # draw the canvas, cache the renderer
        #     image_from_plot = np.frombuffer(h1.canvas.tostring_rgb(), dtype=np.uint8)
        #     image_from_plot = image_from_plot.reshape(h1.canvas.get_width_height()[::-1] + (3,))
        #     img = Image.fromarray(image_from_plot, 'RGB')
        #     gif_images.append(img)
        #     plt.pause(0.01)
        # imageio.mimsave(outfilename, gif_images, fps=10)
        # plt.figure()

        loss = forward()
        error = loss / torch.norm(sample['uT'])
        print('epoch {}, error {}.'.format(epoch, error))
        callback.save(xopt, 'final')

