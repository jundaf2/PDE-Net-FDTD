# %%
import numpy as np
from numpy import *
import numpy.fft as fft
import torch
import torch.utils.data
# %% torch dataset transform tools


def periodicpad(inputs, pad):
    """
    'periodic' pad, similar to torch.nn.functional.pad
    周期填充边界
    """
    # print(pad)
    n = len(inputs.shape)
    inputs = transpose(inputs, list(range(n-1,-1,-1))) # 维度反转
    pad = iter(pad) # pad width
    i = 0  # i的大小是受限于维度dim，目的是对空间域进行周期padding
    indx = []
    for a in pad:
        b = next(pad)
        # print(a,b)
        permute = list(range(n))
        permute[i] = 0
        permute[0] = i
        inputs = transpose(inputs,permute)  # 把第i个dim转换到第0个dim，便于后边操作
        inputlist = [inputs,]
        if a > 0:
            inputlist = [inputs[slice(-a,None)],inputs]  # 最后a个放到最前
        if b > 0:
            inputlist = inputlist+[inputs[slice(0,b)],]  # 前b个放到最后
        if a+b > 0:
            inputs = concatenate(inputlist, axis=0)  # 合并
        inputs = transpose(inputs, permute)
        i += 1
    inputs = transpose(inputs, list(range(n-1,-1,-1)))  # 回归原有顺序
    return inputs


class DownSample(object):
    def __init__(self, scale, boundary='Periodic'):
        assert isinstance(scale, int)
        self.scale = scale
        self.boundary = boundary

    def __call__(self, sample):
        if self.boundary == 'Periodic':
            idx1 = slice(random.randint(self.scale), None, self.scale)
            idx2 = slice(random.randint(self.scale), None, self.scale)
        else:
            idx1 = slice(self.scale - 1, None, self.scale)
            idx2 = slice(self.scale - 1, None, self.scale)
        s = {}
        for k in sample:
            s[k] = sample[k][idx1, idx2]
        return s


class ToTensor(object):
    def __call__(self, sample):
        s = {}
        for k in sample:
            s[k] = torch.from_numpy(sample[k])
        return s


# 用来包裹张量并记录应用的操作。Variable可以看作是对Tensor对象周围的一个薄包装，也包含了和张量相关的梯度，以及对创建它的函数的引用
class ToVariable(object):
    def __call__(self, sample):
        # print("... To torch Variable ...")
        s = {}
        for k in sample:
            s[k] = torch.autograd.Variable(sample[k])
            # print(k,s[k].dtype)
        return s


class ToDevice(object):
    def __init__(self, device):
        assert isinstance(device, int)
        self.device = device

    def __call__(self, sample):
        s = {}
        for k in sample:
            if self.device >= 0:
                s[k] = sample[k].cuda(self.device)
            else:
                s[k] = sample[k].cpu()
        return s


class ToPrecision(object):
    def __init__(self, precision):
        assert precision in ['float', 'double']
        self.precision = precision

    def __call__(self, sample):
        s = {}
        for k in sample:
            if self.precision == 'float':
                s[k] = sample[k].float()
            else:
                s[k] = sample[k].double()
        return s

class AddNoise(object):
    def __init__(self, start_noise_level=0, end_noise_level=0):
        self.start_noise_level = start_noise_level
        self.end_noise_level = end_noise_level

    def __call__(self, sample):
        s = {}
        for k in sample:
            s[k] = sample[k]
        mean = sample['uT'].mean()
        stdvar = sqrt(((sample['uT'] - mean) ** 2).mean())
        size = sample['u0'].size()
        startnoise = sample['u0'].new(size).normal_()
        s['u0'] = sample['u0'] + self.start_noise_level * stdvar * startnoise
        s['u1'] = sample['u1'] + self.start_noise_level * stdvar * startnoise
        if 'uT' in sample:
            s['uT_clean'] = sample['uT']
            size = sample['uT'].size()
            endnoise = sample['uT'].new(size).normal_()
            s['uT'] = sample['uT'] + self.end_noise_level * stdvar * endnoise
        return s

class PDESolver(object):
    def step(self, u1, u0, j2, j0):
        raise NotImplementedError
    def initgen(self):
        raise NotImplementedError
    def predict(self, u1, u0, j2, j0, t):
        assert hasattr(self, 'dt')
        N = int(ceil(t/self.dt))
        for i in range(0, N):
            u1, u0, j2, j0 = self.step(u1, u0, j2, j0)

        return u1, u0

# %% numpy pde data generator
class _fdm2d(PDESolver):

    def __init__(self, freq=1e6, mesh_size=(50,50), boundary='Dirichlet', obj_type = "circle"):
        self.mesh_size = list(mesh_size)
        self.freq = freq
        self.obj_type =obj_type
        assert boundary.upper() in ['Dirichlet'.upper(), 'pml'.upper(), 'abc'.upper(), 'periodic'.upper()]
        from parameters import _parameters
        self.epsr = _parameters.VACUUM_PERMITIVITY*ones(mesh_size)
        self.sigma = zeros(mesh_size)
        self.mu = _parameters.VACUUM_PERMEABILITY

        self.dx = _parameters.DELTA_X
        self.dt = _parameters.DELTA_T
        # print("... time step size {} ...".format(self.dt))
        # print("... spacial grid size {} ...".format(self.dx))
        # print(self.dx , self.dt)
        # print("... mesh size {} ...".format(self.mesh_size))
        self.source_mode = 2 * np.pi *freq*(-1 + 1j)
        self.boundary = boundary.upper()
        self.source_type = 'point'



    def step(self, u1, u0, j2, j0):
        # print('u1 energy (norm): ', linalg.norm(u1))
        # print('u0 energy (norm): ', linalg.norm(u0))
        # print('j2 energy (norm): ', linalg.norm(real(j2)))
        # print('j0 energy (norm): ', linalg.norm(real(j0)))

        assert (u1.shape==u0.shape)
        assert self.source_type == 'point'
        if self.boundary=='Dirichlet'.upper():
            u1 = pad(u1, pad_width=1, mode='constant')
            u0 = pad(u0, pad_width=1, mode='constant')
            # FDM for ME Electric field
            u2 = 2 * u1[1:-1, 1:-1] * (self.mu * self.epsr / self.dt ** 2 - 2 / self.dx ** 2) \
                 + (u1[2:, 1:-1] + u1[:-2, 1:-1]) / self.dx ** 2 \
                 + (u1[1:-1, 2:] + u1[1:-1, :-2]) / self.dx ** 2
            u2 += u0[1:-1, 1:-1] * (self.mu * self.sigma / (2 * self.dt) - self.mu * self.epsr / self.dt ** 2)
            u2 -= self.mu / (2 * self.dt) * (real(j2) - real(j0))
            u2 *= (self.mu * self.sigma / (2 * self.dt) + self.mu * self.epsr / self.dt ** 2) ** (-1)
            # print(self.mu * self.epsr / self.dt ** 2, (self.mu * self.sigma / (2 * self.dt) + self.mu * self.epsr / self.dt ** 2) ** (-1))
            u1 = u1[1:-1, 1:-1]
        elif self.boundary=='periodic'.upper():
            u1 = squeeze(periodicpad(u1[newaxis, newaxis, :, :], [1] * 4))
            u0 = squeeze(periodicpad(u0[newaxis, newaxis, :, :], [1] * 4))

            u2 = 2 * u1[1:-1, 1:-1] * (self.mu * self.epsr / self.dt ** 2 - 2 / self.dx ** 2) \
                 + (u1[2:, 1:-1] + u1[:-2, 1:-1]) / self.dx ** 2 \
                 + (u1[1:-1, 2:] + u1[1:-1, :-2]) / self.dx ** 2
            u2 += u0[1:-1, 1:-1] * (self.mu * self.sigma / (2 * self.dt) - self.mu * self.epsr / self.dt ** 2)
            u2 -= self.mu / (2 * self.dt) * (real(j2) - real(j0))
            u2 *= (self.mu * self.sigma / (2 * self.dt) + self.mu * self.epsr / self.dt ** 2) ** (-1)
            # print(self.mu * self.epsr / self.dt ** 2, (self.mu * self.sigma / (2 * self.dt) + self.mu * self.epsr / self.dt ** 2) ** (-1))
            u1 = u1[1:-1, 1:-1]
        elif self.boundary=='abc'.upper():
            raise NotImplementedError
        elif self.boundary=='pml'.upper():
            raise NotImplementedError
        j0 = j0 + self.dt * self.source_mode * j0
        j2 = j2 + self.dt * self.source_mode * j2

        return u2, u1, j2, j0

    def initgen(self):
        assert hasattr(self, 'dt')
        object_locationx = self.mesh_size[0] // 4 #
        object_locationy = self.mesh_size[1] // 4 #
        object_radius = random.randint(2, self.mesh_size[0] // 4)

        self.src_locx = self.mesh_size[0]//2
        self.src_locy = self.mesh_size[1]//2
        # print('source location:', (self.src_locx, self.src_locy))

        epsr = 1. * random.randint(2, 10)
        sigma = 1e9
        if self.obj_type == "circle":
            # Circle
            for y in range(self.mesh_size[0]):
                for x in range(self.mesh_size[1]):
                    ydist = (object_locationy - y)
                    xdist = (object_locationx - x)
                    dist = np.sqrt(xdist ** 2 + ydist**2)
                    if dist <= object_radius:
                        # lossy material
                        self.epsr[y, x] *= epsr
                        self.sigma[y, x] = sigma
        elif self.obj_type == "square":
            # Square
            for y in range(self.mesh_size[0]):
                for x in range(self.mesh_size[0]):
                    if ((object_locationy - object_radius) < y) and (y < (object_locationy + object_radius)) \
                            and ((object_locationx - object_radius) < x) and (x < (object_locationx + object_radius)):
                        # lossy material
                        self.epsr[y, x] *= epsr
                        self.sigma[y, x] = sigma
        # Mode problem for source J
        j0 = zeros(self.mesh_size, dtype=complex)
        j0[self.src_locx, self.src_locy] = 1  # initial magnitude
        j1 = j0 + self.dt*self.source_mode*j0
        j2 = j1 + self.dt*self.source_mode*j1
        u1 = zeros(self.mesh_size)
        u0 = zeros(self.mesh_size)

        return u1, u0, j2, j0, self.epsr, self.sigma

# %% torch pde dataset
# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class TorchPDEDataSet(torch.utils.data.Dataset):
    def _xy(self): # 构造向量x作为行/列向量的矩阵, 类似 meshgrid, 用于interpolation
        x = self.pde.dx * arange(self.mesh_size[0])
        sample = {}
        if self.boundary.upper() == 'Dirichlet'.upper():
            sample['x'] = repeat(x[newaxis, :], self.mesh_size[0], axis=0)
            sample['y'] = repeat(x[:, newaxis], self.mesh_size[0], axis=1)
        else:
            x = x[1:]
            sample['x'] = repeat(x[newaxis, :], self.mesh_size[0] - 1, axis=0)
            sample['y'] = repeat(x[:, newaxis], self.mesh_size[0] - 1, axis=1)
        return sample

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        fdm2d = _fdm2d(freq=self.freq, mesh_size=self.mesh_size, boundary=self.boundary)
        u1, u0, j2, j0, espr, sigma = fdm2d.initgen()
        sample = {}
        sample['u0'] = u0
        sample['u1'] = u1
        sample['epsr'] = espr
        sample['sigma'] = sigma
        if isinstance(self.T, float):
            ut1, ut0 = fdm2d.predict(u1, u0, j2, j0, self.T)
        sample['uT'] = ut1
        return sample

class fdm2d(TorchPDEDataSet):
    def __init__(self, T, mesh_size, freq=5e9, transform=None, boundary='Dirichlet'):
        assert boundary.upper() in ['Dirichlet'.upper(), 'pml'.upper(), 'abc'.upper(), 'periodic'.upper()]
        if isinstance(mesh_size, int):
            self.mesh_size = [mesh_size, ] * 2
        else:
            assert mesh_size[0] == mesh_size[1]
            self.mesh_size = mesh_size[:2]
        self.freq = freq
        self.mesh_size = mesh_size
        self.T = T
        self.boundary = boundary
        self.transform = transform
        self.size = 100


def test_fdm2d():
    import matplotlib.pyplot as plt
    h = plt.figure()
    a = h.add_subplot(111)
    fdm2d = _fdm2d(freq=1e6, mesh_size=(50, 50), boundary='periodic')
    u1, u0, j2, j0, epsr, sigma = fdm2d.initgen()
    for i in range(1, 100):
        u1, u0, j2, j0 = fdm2d.step(u1, u0, j2, j0)
        print('domain energy (norm): ', linalg.norm(u1))
        a.clear()
        b = a.imshow(u1, cmap='jet')
        a.set_title('t={:.2f}'.format(i))
        c = h.colorbar(b, ax=a)
        plt.pause(1e-2)
        c.remove()
    c = h.colorbar(b, ax=a)

# import time
# starttime = time.time()
# test_fdm2d()
# endtime = time.time()
# print('总共的时间为:', round(endtime - starttime, 2),'secs')

def test_dataset():
    import torchvision  # 用Compose把多个步骤整合到一起
    trans = None
    d = fdm2d(T=[2e-9/4, 2e-9/4*2, 2e-9/4*3, 2e-9], time_range=2e-9, mesh_size=(50, 50), boundary='Dirichlet')
    dataloader = torch.utils.data.DataLoader(d, batch_size=1, num_workers=0)
    dataloader = iter(dataloader)
    sample = next(dataloader)
    for i, input in enumerate(dataloader):
        print(i, type(input))
# test_dataset()
