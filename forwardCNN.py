from numpy import *
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import parameters
import forward_cnn_pdedata as pdedata
import FD
import NFI
def j_gen(step, freq, dim, dt):
    source_mode = -freq * 2 + 2 * pi * freq * 1j
    mesh_size = [dim, ] * 2
    src_locx = mesh_size[0] // 2# + random.randint(- mesh_size[0] // 3, mesh_size[0] // 3)
    src_locy = mesh_size[0] // 2# + random.randint(- mesh_size[0] // 3, mesh_size[0] // 3)
    print('source location:', (src_locx, src_locy))
    j0 = zeros(mesh_size, dtype=complex)

    j0[src_locx, src_locy] = 1  # initial magnitude
    j1 = j0 + dt * source_mode * j0
    j2 = j1 + dt * source_mode * j1
    for i in range(step):
        # print('step',i)
        yield (torch.FloatTensor(real(j2)).cuda(device=0), torch.FloatTensor(real(j0)).cuda(device=0))
        j0 = j0 + dt * source_mode * j0
        j2 = j2 + dt * source_mode * j2

def net_info(net):
    print(type(net.parameters()))
    for p in list(net.parameters()):
        print(p)
    print(list(net.parameters())[0].device)
class CnnFdmNet(torch.nn.Module):
    def __init__(self, kernel_size, max_order, dx, dt, constraint):
        super(CnnFdmNet, self).__init__()
        self.kernel_size=kernel_size
        self.register_buffer('dt', torch.FloatTensor(1).fill_(parameters._parameters.DELTA_T))
        self.register_buffer('c', torch.FloatTensor(1).fill_(parameters._parameters.SPEED_LIGHT))
        self.register_buffer('epsilon', torch.FloatTensor(1).fill_(parameters._parameters.VACUUM_PERMITIVITY))
        self.dt = Variable(self.dt).cuda(device=0)
        self.c = Variable(self.c).cuda(device=0)
        self.epsilon = Variable(self.epsilon).cuda(device=0)

        dx=parameters._parameters.DELTA_X
        D02 = array([[0, 0, 0], [1 / dx ** 2, -2 / dx ** 2, 1 / dx ** 2], [0, 0, 0]])
        D20 = transpose(D02)
        D02 = torch.FloatTensor(D02).unsqueeze(0).unsqueeze(0)
        D20 = torch.FloatTensor(D20).unsqueeze(0).unsqueeze(0)
        self.D02 = torch.nn.Parameter(data=D02, requires_grad=False)#.cuda(device=0)
        self.D20 = torch.nn.Parameter(data=D20, requires_grad=False)#.cuda(device=0)

        # print(self.D02,self.D20)

    def forward(self, init):
        u1 = init[0]
        u0 = init[1]
        j2 = init[2]
        j0 = init[3]
        # print("... network input data dimension is {} {} {} {}...".format(u1.size(), u0.size(), j2.size(), j0.size()))

        D20u1 = torch.nn.functional.conv2d(u1, self.D20, padding=[self.kernel_size // 2]*2, stride=1)
        D02u1 = torch.nn.functional.conv2d(u1, self.D02, padding=[self.kernel_size // 2]*2, stride=1)
        # 如果对在GPU上的数据进行运算，那么结果还是存放在GPU上
        # 在gpu上做运算．通过.cuda()将模型计算放到gpu.相应的,传给模型的输入也必须是gpu显存上的数据
        # print(self.dt ** 2 * self.c ** 2, self.dt / (2 * self.epsilon) )
        u2 = 2 * u1 - u0 + self.dt ** 2 * self.c ** 2 *(D20u1 + D02u1) - self.dt / (2 * self.epsilon) * (j2 - j0)

        return u2

    def step(self, init, jgen, stepnum):
        print("... cnn predicting future {} steps ...".format(stepnum))
        u1 = init[0]
        u0 = init[1]
        for t in range(stepnum):
            j = next(jgen)
            j2 = j[0][newaxis, newaxis, :, :]
            j0 = j[1][newaxis, newaxis, :, :]
            initial_data = (u1, u0, j2, j0)
            u0 = u1
            u2 = self.forward(initial_data)
            u1 = u2
        return u1
dim = 4000
dt = parameters._parameters.DELTA_T
dt_cnn = dt
dx = parameters._parameters.DELTA_X
step_fdm = 100
step_cnn = 100
freq = parameters._parameters.FREQUENCY

d = pdedata.fdm2d(T=step_fdm * dt, mesh_size=(dim, dim), freq=freq)
dataloader = torch.utils.data.DataLoader(d, batch_size=1, num_workers=0)
dataloader = iter(dataloader)
sample = pdedata.ToVariable()(pdedata.ToDevice(device=0)(pdedata.ToPrecision('float')(next(dataloader))))
del dataloader

# generate souce information for a longer time step: step_cnn
jgen_cnn = j_gen(step_cnn, freq, dim, dt)
jgen_fdm = j_gen(step_fdm+1, freq, dim, dt)
initial_data = (sample['u1'][:, newaxis, :, :], sample['u0'][:, newaxis, :, :])

cnnfdmnet = CnnFdmNet(kernel_size=3, max_order=2, dx=dx, dt=dt_cnn, constraint='DIRICHLET').float()
cnnfdmnet.cuda(device=0)


u0 = sample['u0'][:, newaxis, :, :]
u1 = sample['u1'][:, newaxis, :, :]
print("... cnn predicting future {} steps ...".format(step_cnn))
import time
starttime = time.time()
for t in range(step_cnn):
    if t<=20:
        j = next(jgen_cnn)
        j2 = j[0][newaxis, newaxis, :, :]
        j0 = j[1][newaxis, newaxis, :, :]
    else:
        j2 = 0
        j0 = 0
    initial_data = (u1, u0, j2, j0)
    u0 = u1
    u2 = cnnfdmnet(initial_data)
    u1 = u2
    # a.clear()
    # print('domain energy (norm): ', linalg.norm(u1.squeeze().cpu().data))
    # b = a.imshow(u1.squeeze().cpu().data)#, cmap='jet')
    # a.set_title('t={:f}s'.format(t*dt_cnn))
    # c = h.colorbar(b, ax=a)
    # plt.pause(1e-2)
    # c.remove()

endtime = time.time()
print("... cnn predicting future {} steps ...".format(step_cnn))
print('总共的时间为:', round(endtime - starttime, 2),'secs')
difference = linalg.norm((u1.squeeze() - sample['uT'].squeeze()).cpu().data)
print('energy：', linalg.norm((u1.squeeze().cpu().data)))
print('error：', difference)
