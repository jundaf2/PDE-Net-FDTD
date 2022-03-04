# %%
import numpy as np
from numpy import *
import numpy.fft as fft
import torch
import torch.utils.data

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
    def __init__(self, start_noise_level, end_noise_level):
        self.start_noise_level = start_noise_level
        self.end_noise_level = end_noise_level

    def __call__(self, sample):
        s = {}
        for k in sample:
            s[k] = sample[k]
        mean = sample['u0'].mean()
        stdvar = sqrt(((sample['u0'] - mean) ** 2).mean())
        size = sample['u0'].size()
        startnoise = sample['u0'].new(size).normal_()
        s['u0'] = sample['u0'] + self.start_noise_level * stdvar * startnoise
        if 'uT' in sample:
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
        print("... fdm predicting future {} steps ...".format(N))
        import time
        starttime = time.time()

        for i in range(0, N):
            if i<=20:
                u1, u0, j2, j0 = self.step(u1, u0, j2, j0)
            else:
                u1, u0, _, _ = self.step(u1, u0, 0, 0)

        endtime = time.time()
        print("... fdm predicted future {} steps ...".format(N))
        print('总共的时间为:', round(endtime - starttime, 2), 'secs')
        return u1, u0


class _fdm2d(PDESolver):

    def __init__(self, freq=1e6, mesh_size=(50,50), boundary='Dirichlet'):
        assert boundary.upper() in ['Dirichlet'.upper(), 'pml'.upper()]
        from parameters import _parameters
        self.epsr = _parameters.VACUUM_PERMITIVITY
        self.sigma = 0
        self.mu = _parameters.VACUUM_PERMEABILITY
        c = _parameters.SPEED_LIGHT
        self.dx = _parameters.DELTA_X
        self.dt = _parameters.DELTA_T
        # print("... time step size {} ...".format(self.dt))
        # print("... spacial grid size {} ...".format(self.dx))
        # print(self.dx , self.dt)
        self.mesh_size = list(mesh_size)
        # print("... mesh size {} ...".format(self.mesh_size))
        self.freq = freq
        self.source_mode = -freq * 2 + 2*np.pi*freq * 1j
        self.boundary = boundary.upper()
        self.source_type = 'point'

    def step(self, u1, u0, j2, j0):
        # print('u1 energy (norm): ', linalg.norm(u1))
        # print('u0 energy (norm): ', linalg.norm(u0))
        # print('j2 energy (norm): ', linalg.norm(real(j2)))
        # print('j0 energy (norm): ', linalg.norm(real(j0)))
        assert (u1.shape==u0.shape)
        if self.boundary=='Dirichlet'.upper():
            u1 = pad(u1, pad_width=1, mode='constant')
            u0 = pad(u0, pad_width=1, mode='constant')

        elif self.boundary=='pml'.upper():
            raise NotImplementedError
        assert self.source_type == 'point'
        # FDM for ME Electric field
        u2 = 2 * u1[1:-1, 1:-1] * (self.mu * self.epsr / self.dt ** 2 - 2 / self.dx ** 2)\
                                    + (u1[2:, 1:-1] + u1[:-2, 1:-1]) / self.dx ** 2 \
                                    + (u1[1:-1, 2:] + u1[1:-1, :-2]) / self.dx ** 2
        u2 += u0[1:-1, 1:-1] * (self.mu * self.sigma / (2 * self.dt) - self.mu * self.epsr / self.dt ** 2)
        u2 -= self.mu / (2 * self.dt) * (real(j2) - real(j0))
        u2 *= (self.mu * self.sigma / (2 * self.dt) + self.mu * self.epsr / self.dt ** 2) ** (-1)
        # print(self.mu * self.epsr / self.dt ** 2, (self.mu * self.sigma / (2 * self.dt) + self.mu * self.epsr / self.dt ** 2) ** (-1))
        u1 = u1[1:-1, 1:-1]
        j0 = j0 + self.dt * self.source_mode * j0
        j2 = j2 + self.dt * self.source_mode * j2

        return u2, u1, j2, j0

    def initgen(self):
        assert hasattr(self, 'dt')
        self.src_locx = self.mesh_size[0] // 2 #+ random.randint(-self.mesh_size[0] // 3, self.mesh_size[0] // 3)
        self.src_locy = self.mesh_size[0] // 2 #+ random.randint(-self.mesh_size[0] // 3, self.mesh_size[0] // 3)
        print('source location:',(self.src_locx,self.src_locy))
        # Mode problem for source J
        j0 = zeros(self.mesh_size, dtype=complex)
        j0[self.src_locx, self.src_locy] = 1  # initial magnitude
        j1 = j0 + self.dt*self.source_mode*j0
        j2 = j1 + self.dt*self.source_mode*j1
        u1 = zeros(self.mesh_size)
        u0 = zeros(self.mesh_size)
        # for i in range(10): u1, u0, j2, j0 = self.step(u1, u0, j2, j0)
        return u1, u0, j2, j0

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
        u1, u0, j2, j0 = self.pde.initgen()
        sample = {}
        sample['u0'] = u0
        sample['u1'] = u1

        if isinstance(self.T, float): #未来某个时间
            #ut1, ut0, jt2, jt0 = self.pde.predict(u1, u0, j2, j0, self.T)
            ut1, ut0 = self.pde.predict(u1, u0, j2, j0, self.T)
        else: #未来多个时间
            assert isinstance(self.T[0], float)
            n = len(self.T)
            ut1 = np.zeros([n, ] + list(u1.shape))
            ut0 = np.zeros([n, ] + list(u0.shape))
            jt2 = np.zeros([n, ] + list(j2.shape))
            jt0 = np.zeros([n, ] + list(j0.shape))
            T = [0, ] + list(self.T)
            for i in range(n):
                # print(i)
                u1, u0, j2, j0 = self.pde.predict(u1, u0, j2, j0, T[i + 1] - T[i])
                ut1[i, :, :] = u1
                ut0[i, :, :] = u0
                jt2[i, :, :] = j2
                jt0[i, :, :] = j0
        sample['uT'] = ut1
        # sample['j0set'] = real(jt0)
        # sample['j2set'] = real(jt2)
        # sample['uT0'] = ut0
        # sample['jT2'] = jt2
        # sample['jT0'] = jt0
        # print(not self.transform is None)
        if not self.transform is None:
            # print("... transform sample ...")
            sample = self.transform(sample)
        return sample

class fdm2d(TorchPDEDataSet):
    def __init__(self, T, mesh_size, freq, transform=None, boundary='Dirichlet'):
        assert boundary.upper() in ['Dirichlet'.upper(), 'pml'.upper()]
        if isinstance(mesh_size, int):
            self.mesh_size = [mesh_size, ] * 2
        else:
            assert mesh_size[0] == mesh_size[1]
            self.mesh_size = mesh_size[:2]
        self.T = T
        self.boundary = boundary
        self.pde = _fdm2d(freq=freq, mesh_size=mesh_size, boundary=self.boundary)
        self.transform = transform
        self.size = 100


def test_fdm2d():
    import matplotlib.pyplot as plt
    h = plt.figure()
    a = h.add_subplot(111)
    fdm2d = _fdm2d(freq=1e6, mesh_size=(20, 20), boundary='Dirichlet')
    u1, u0, j2, j0 = fdm2d.initgen()
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
    trans = None #torchvision.transforms.Compose([DownSample(4, boundary='Dirichlet'), ToTensor()])
    d = fdm2d(T=[2e-9/4, 2e-9/4*2, 2e-9/4*3, 2e-9], time_range=2e-9, mesh_size=(50, 50), boundary='Dirichlet')
    dataloader = torch.utils.data.DataLoader(d, batch_size=1, num_workers=0)
    dataloader = iter(dataloader)
    sample = next(dataloader)
    for i, input in enumerate(dataloader):
        print(i, type(input))
# test_dataset()
