#%%
from numpy import *
import torch
from torch.autograd import Variable
import FD
import matplotlib.pyplot as plt
import parameters

def j_gen(step, freq, dim, dt):
    dt = dt.cpu().data.numpy()
    source_mode = -freq * 2 + 2 * pi * freq * 1j
    mesh_size = [dim, ] * 2
    src_locx = mesh_size[0] // 2 #+ random.randint(- mesh_size[0] // 3, mesh_size[0] // 3)
    src_locy = mesh_size[0] // 2 #+ random.randint(- mesh_size[0] // 3, mesh_size[0] // 3)
    print('source location:', (src_locx, src_locy))
    jj0 = zeros(mesh_size, dtype=complex)

    jj0[src_locx, src_locy] = 1  # initial magnitude
    jj1 = jj0 + dt * source_mode * jj0
    jj2 = jj1 + dt * source_mode * jj1
    for i in range(step):
        # print('step',i)
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
        # self.combo.weight.data.uniform_(0,.1)
        self.combofd.weight.data.fill_(0)

        self.dt = torch.tensor(dt)
        self.mu = torch.tensor(parameters._parameters.VACUUM_PERMEABILITY)

        # self.kernel_size=kernel_size[0]
        # D02 = array([[0, 0, 0], [1/dx**2, -2/dx**2, 1/dx**2], [0, 0, 0]])
        # D20 = transpose(D02)
        # D02 = torch.FloatTensor(D02).repeat(1,1,1).unsqueeze(1)
        # D20 = torch.FloatTensor(D20).repeat(1,1,1).unsqueeze(1)
        # self.D02 = torch.nn.Parameter(data=D02, requires_grad=False)
        # self.D20 = torch.nn.Parameter(data=D20, requires_grad=False)

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
        j2 = init[2]
        j0 = init[3]
        epsr = init[4]
        sigma = init[5]
        # print("... network input data dimension is {} ...".format(init[0].dim()))
        # print(epsr.size(), sigma.size())
        # self.plt_moment_kernels(self.fd2d_u1, self.fig1, self.ax1)
        # self.plt_kernel_kernels(self.fd2d_u1.MomentBank, self.fig2, self.ax2)
        # plt.pause(0.1)
        idkernel_u1 = self.id_u1.MomentBank.kernel()
        idkernel_u0 = self.id_u0.MomentBank.kernel()
        fdkernel_u1 = self.fd2d_u1.MomentBank.kernel()

        dt = self.dt
        mu = self.mu
        # jgen = j_gen(stepnum, parameters._parameters.FREQUENCY, parameters._parameters.DIM, self.dt)
        # print("... prediction step number is {} ...".format(stepnum))
        for i in range(stepnum):
            if i>0:
                u0 = u1
                u1 = u2

            # print(j2[:, 0, :, :].size(),j0[:, 0, :, :].size())
            u1id = self.id_u1(u1, idkernel_u1)
            u0id = self.id_u0(u0, idkernel_u0)
            u1fd = self.fd2d_u1(u1, fdkernel_u1)
            # D20u1 = torch.nn.functional.conv2d(u1, self.D20, padding=[self.kernel_size // 2] * 2, stride=1)
            # D02u1 = torch.nn.functional.conv2d(u1, self.D02, padding=[self.kernel_size // 2] * 2, stride=1)

            # print('momentkernel vs cnn:') #already proved equal
            # print('idu1',torch.max(u1id - u1))
            # print('idu0', torch.max(u0id - u0))
            # print(u1fd[:, 3].size(),D20u1.squeeze().size())
            # print('d20',torch.max(u1fd[:, 3] - D20u1.squeeze()))
            # print('d02',torch.max(u1fd[:, 5] - D02u1.squeeze()))
            # print('fdcombo-d02-d20', torch.max(D20u1.squeeze()+D02u1.squeeze()-self.combo(u1fd[:, 1:]).squeeze(1)))
            # jj = next(jgen)
            # print('current j',torch.max((jj[0].repeat(2,1,1)-jj[1].repeat(2,1,1))-(j2[:,i,:,:] - j0[:,i,:,:])))
            # u1fd[:, 1:] 是高阶算子的结果
            # u2 = 2*u1id - u0id\
            #      + self.dt**2 * self.c**2 * self.combo(u1fd) - dt / (2*self.epsilon) * (j2[:,i,:,:] - j0[:,i,:,:]).unsqueeze(1)
            # uFD = self.combofd(u1fd[:, 1:])
            # j_ext = (j2[:,i,:,:]-j0[:,i,:,:]).unsqueeze(1)
            # comboin = torch.cat((u0id,u1id,uFD,j_ext),dim=1)
            # u2 = self.combo(comboin)
            u2=1/(2*epsr/dt+sigma)\
               *(2*dt/mu*self.combofd(u1fd[:, 1:])-1*(j2[:,i,:,:]-j0[:,i,:,:]).unsqueeze(1)+4*epsr/dt*u1id-(2*epsr/dt-sigma)*u0id)

            # u2 = 2 * u1 - u0 + self.dt ** 2 * self.c ** 2 * (D20u1 + D02u1) - self.dt/(2*self.epsilon)*(j2[:,i,:,:]-j0[:,i,:,:]).unsqueeze(1)
        return u2

    def step(self, u):
        return self.forward(u, stepnum=1)