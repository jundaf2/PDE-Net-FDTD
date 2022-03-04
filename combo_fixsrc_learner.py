#%%
from numpy import *
import torch
from torch.autograd import Variable
import FD
import matplotlib.pyplot as plt
import parameters

def j_gen(step, freq, dim, dt):
    dt = dt.cpu().data.numpy()
    source_mode = 2 * pi * freq * (-1 + 1j)

    mesh_size = [dim, ] * 2
    src_locx = mesh_size[0] // 2
    src_locy = mesh_size[1] // 2

    # print('source location:', (src_locx, src_locy))
    jj0 = zeros(mesh_size, dtype=complex)

    jj0[src_locx, src_locy] = 1  # initial magnitude
    jj1 = jj0 + dt * source_mode * jj0
    jj2 = jj1 + dt * source_mode * jj1
    for i in range(step):
        yield (torch.FloatTensor(real(jj2)).cuda(device=0), torch.FloatTensor(real(jj0)).cuda(device=0))
        jj0 = jj0 + dt * source_mode * jj0
        jj2 = jj2 + dt * source_mode * jj2

class SingleNonLinear2d(torch.nn.Module): # xy是mesh
    # The filters associated to the convolution operators fD0;Dij :  i + j <= 4
    # and the coefficients of the piecewise quadratic polynomials
    # are the trainable parameters of the network.
    def __init__(self, kernel_size, max_order, dx, dt, constraint, boudary='Dirichlet'):
        super(SingleNonLinear2d, self).__init__()
        self.id_u1 = FD.FD2d(1, 0, dx=dx, constraint=constraint, boundary=boudary)
        self.id_u0 = FD.FD2d(1, 0, dx=dx, constraint=constraint, boundary=boudary)
        self.fd2d_u1 = FD.FD2d(kernel_size, max_order, dx=dx, constraint=constraint, boundary=boudary)
        derivative_num = sum(arange(self.fd2d_u1.MomentBank.max_num_kernel_each_order)+1)-1
        self.combofd = torch.nn.Conv2d(in_channels=derivative_num, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.combofd.weight.data.fill_(0)

        self.dt = torch.tensor(dt)
        self.mu = torch.tensor(parameters._parameters.VACUUM_PERMEABILITY)

        # plt.figure(3)
        # self.fig1, self.ax1 = plt.subplots(nrows=max_order + 1, ncols=self.fd2d_u1.MomentBank.max_num_kernel_each_order
        #                        , sharex='col', sharey='row')
        # plt.figure(4)
        # self.fig2, self.ax2 = plt.subplots(nrows=max_order + 1, ncols=self.fd2d_u1.MomentBank.max_num_kernel_each_order
        #                                    , sharex='col', sharey='row')


    def diff_params(self):
        return list(self.combofd.parameters()) #+ list(self.fd2d_u1.parameters()) ##+list(self.id_u1.parameters()) + list(self.id_u0.parameters())


    # module.parameter() is the moment --> self.moment = nn.Parameter(moment) in FD
    def plt_moment_kernels(self, kernel, fig, ax):
        if hasattr(ax,"flatten"):
            axes = ax.flatten()
            i = 0
            parameters = list(kernel.parameters())
            # print(parameters[0])
            for a in axes:
                a.clear()
            for order, order_list in enumerate(kernel.MomentBank._order_bank):
                for no_order_pair, order_pair in enumerate(order_list):
                    i += 1
                    axes[order * kernel.MomentBank.max_num_kernel_each_order + no_order_pair].matshow(
                        parameters[0][i - 1].detach().cpu().data)
                    axes[order * kernel.MomentBank.max_num_kernel_each_order + no_order_pair].set_title(
                        "$D{}{}$".format(order_pair[0], order_pair[1]))
            for a in axes:
                a.axis('off')
        else:
            axes=ax
            parameters = list(kernel.parameters())
            axes.clear()
            axes.matshow(parameters[0][0].detach().cpu().data)
            axes.axis('off')

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    def plt_kernel_kernels(self, kernel, fig, ax):
        if hasattr(ax, "flatten"):
            axes = ax.flatten()
            i = 0
            parameters = kernel.kernel()
            # print(parameters.size())
            for a in axes:
                a.clear()
            for order, order_list in enumerate(kernel._order_bank):
                for no_order_pair, order_pair in enumerate(order_list):
                    i += 1
                    axes[order * kernel.max_num_kernel_each_order + no_order_pair].matshow(
                        parameters[i - 1][0].detach().cpu().data)
                    axes[order * kernel.max_num_kernel_each_order + no_order_pair].set_title(
                        "$D{}{}$".format(order_pair[0], order_pair[1]))
            for a in axes:
                a.axis('off')
        else:
            axes = ax
            parameters = kernel.kernel()
            axes.clear()
            axes.matshow(parameters[0][0].detach().cpu().data)
            axes.axis('off')

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        # plt.show()

    def forward(self, init, stepnum):
        assert init[0].dim() == 4
        u1 = init[0]
        u0 = init[1]
        epsr = init[2]
        sigma = init[3]
        batch = parameters._parameters.BATCH

        idkernel_u1 = self.id_u1.MomentBank.kernel()
        idkernel_u0 = self.id_u0.MomentBank.kernel()
        fdkernel_u1 = self.fd2d_u1.MomentBank.kernel()

        dt = self.dt
        mu = self.mu
        jgen = j_gen(stepnum, parameters._parameters.FREQUENCY, parameters._parameters.DIM, self.dt)

        for i in range(stepnum):
            if i>0:
                u0 = u1
                u1 = u2
            u1id = self.id_u1(u1, idkernel_u1)
            u0id = self.id_u0(u0, idkernel_u0)
            u1fd = self.fd2d_u1(u1, fdkernel_u1)
            jj = next(jgen)
            u2 = 1/(2*epsr/dt+sigma)\
                *(2*dt/mu*self.combofd(u1fd[:, 1:])-1*(jj[0].repeat(batch,1,1)-jj[1].repeat(batch,1,1)).unsqueeze(1)+4*epsr/dt*u1id-(2*epsr/dt-sigma)*u0id)
        return u2

    def step(self, u):
        return self.forward(u, stepnum=1)