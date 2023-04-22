import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from celluloid import Camera


# from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.utils import init
from dmp.utils.dmp_layer import DMPIntegrator, DMPParameters
# from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr import pytorch_util as ptu
import cv2
# from dmp.utils.smnist_loader import MatLoader
# from os.path import dirname, realpath
import os
# import sys
import argparse
from datetime import datetime
# from dmp.utils.mnist_cnn import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='ndp-il')
args = parser.parse_args()




import scipy.io as sio
import torch
from torch.autograd import Variable
import numpy as np


class Mapping:
    y_max = 1
    y_min = -1
    x_max = []
    x_min = []


class MatLoader:
    def load_data(file,
                  load_original_trajectories=False,
                  image_key='imageArray',
                  traj_key='trajArray',
                  dmp_params_key='DMPParamsArray',
                  dmp_traj_key='DMPTrajArray'):
        # Load data struct

        # file = Set file variable to /path/to/arr.m 
        data = sio.loadmat(file)

        # Parse data struct
        if 'Data' in data:
            data = data['Data']
        # Backward compatibility with old format
        elif 'slike' in data:
            data = data['slike']
            image_key = 'im'
            traj_key = 'trj'
            dmp_params_key = 'DMP_object'
            dmp_traj_key = 'DMP_trj'

        # Load images
        images = []
        for image in data[image_key][0]:
            images.append(image.astype('float'))
        images = np.array(images)

        # Load DMPs
        DMP_data = data[dmp_params_key][0]
        outputs = []
        for dmp in DMP_data:
            tau = dmp['tau'][0, 0][0, 0]
            w = dmp['w'][0, 0]
            goal = dmp['goal'][0, 0][0]
            y0 = dmp['y0'][0, 0][0]
            # dy0 = np.array([0,0])
            learn = np.append(tau, y0)
            # learn = np.append(learn,dy0)
            learn = np.append(learn, goal)  # Correction
            learn = np.append(learn, w)
            outputs.append(learn)
        outputs = np.array(outputs)
        '''
        scale = np.array([np.abs(outputs[:, i]).max() for i in range(0, 5)])
        scale = np.concatenate((scale, np.array([np.abs(outputs[:, 5:outputs.shape[1]]).max() for i in range(5, outputs.shape[1])])))
        '''

        # Scale outputs
        y_max = 1
        y_min = -1
        x_max = np.array([outputs[:, i].max() for i in range(0, 5)])
        x_max = np.concatenate(
            (x_max, np.array([outputs[:, 5:outputs.shape[1]].max() for i in range(5, outputs.shape[1])])))
        x_min = np.array([outputs[:, i].min() for i in range(0, 5)])
        x_min = np.concatenate(
            (x_min, np.array([outputs[:, 5:outputs.shape[1]].min() for i in range(5, outputs.shape[1])])))
        scale = x_max-x_min
        scale[np.where(scale == 0)] = 1
        outputs = (y_max - y_min) * (outputs-x_min) / scale + y_min

        # Load scaling
        scaling = Mapping()
        scaling.x_max = x_max
        scaling.x_min = x_min
        scaling.y_max = y_max
        scaling.y_min = y_min

        # Load original trajectories
        original_trj = []
        if load_original_trajectories:
            trj_data = data[traj_key][0]
            original_trj = [(trj) for trj in trj_data[:]]

        return images, outputs, scaling, original_trj

    def data_for_network(images, outputs):
        input_data = Variable(torch.from_numpy(images)).float()
        output_data = Variable(torch.from_numpy(outputs), requires_grad=False).float()
        return input_data, output_data



def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




class NdpCnn(nn.Module):
    def __init__(
            self,
            init_w=3e-3,
            layer_sizes=[784, 200, 100],
            hidden_activation=F.relu,
            pt='./dmp/data/pretrained_weights.pt',
            output_activation=None,
            hidden_init=fanin_init,
            b_init_value=0.1,
            state_index=np.arange(1),
            N = 5,
            T = 10,
            l = 10,
            *args,
            **kwargs
    ):
        super().__init__()
        self.N = N
        self.l = l
        self.output_size = N*len(state_index) + 2*len(state_index)
        output_size = self.output_size
        self.T = T
        self.output_activation=torch.tanh
        self.state_index = state_index
        self.output_dim = output_size
        tau = 1
        dt = 1.0 / (T*self.l)
        self.output_activation=torch.tanh
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None)
        self.func = DMPIntegrator()
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)
        layer_sizes=[784, 200, 100, 200, 2*output_size, output_size]
        self.input_size = input_size
        self.hidden_activation = hidden_activation
        in_size = input_size
        self.pt = CNN()
        self.pt.load_state_dict(torch.load(pt))
        self.convSize = 4*4*50
        self.imageSize = 28
        self.N = N
        self.middle_layers = []
        for i in range(1, len(layer_sizes)-1):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            hidden_init(layer.weight)
            layer.bias.data.fill_(b_init_value)
            self.middle_layers.append(layer)
            self.add_module("middle_layer_" + str(i), layer)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.last_fc = init_(nn.Linear(layer_sizes[-1], output_size))



    def forward(self, input, y0, return_preactivations=False):
        x = input
        x = x.view(-1, 1, self.imageSize, self.imageSize)
        x = F.relu(self.pt.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu((self.pt.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        activation_fn = self.hidden_activation
        x = activation_fn(self.pt.fc1(x))
        for layer in self.middle_layers:
            x = activation_fn(layer(x))
        output = self.last_fc(x)*1000
        y0 = y0.reshape(-1, 1)[:, 0]
        dy0 = torch.zeros_like(y0) + 0.01
        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0)
        y = y.view(input.shape[0], len(self.state_index), -1)
        return y.transpose(2, 1)





data_path = '/home/mateo/Downloads/arr.m'
images, outputs, scale, or_tr = MatLoader.load_data(data_path, load_original_trajectories=True)


# test_trajectory = or_tr[8]
# all_x = []
# all_y = []
# fig = plt.figure()
# camera = Camera(fig)
# for i in range(0, test_trajectory.shape[0]):
#     x, y = test_trajectory[i][0], test_trajectory[i][1]
#     all_x.append(x)
#     all_y.append(y)
#     plt.scatter(all_x, all_y)
#     # plt.show()
#     camera.snap()
# animation = camera.animate()
# animation.save('animation.mp4')
# import pdb;pdb.set_trace()

images = np.array([cv2.resize(img, (28, 28)) for img in images])/255.0
input_size = images.shape[1] * images.shape[2]
import pdb;pdb.set_trace()

inds = np.arange(12000)
np.random.shuffle(inds)
test_inds = inds[10000:]
train_inds = inds[:10000]
X = torch.Tensor(images[:12000]).float()
Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:12000]


time = str(datetime.now())
time = time.replace(' ', '_')
time = time.replace(':', '_')
time = time.replace('-', '_')
time = time.replace('.', '_')
model_save_path = './dmp/data/' + args.name + '_' + time
os.mkdir(model_save_path)
k = 1
T = 300/k
N = 30
learning_rate = 1e-3
Y = Y[:, ::k, :]


X_train = X[train_inds]
Y_train = Y[train_inds]
X_test = X[test_inds]
Y_test = Y[test_inds]



num_epochs = 71
batch_size = 100
ndpn = NdpCnn(T=T,l=1, N=N, state_index=np.arange(2))
optimizer = torch.optim.Adam(ndpn.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    inds = np.arange(X_train.shape[0])
    np.random.shuffle(inds)
    for ind in np.split(inds, len(inds)//batch_size):
        optimizer.zero_grad()
        y_h = ndpn(X_train[ind], Y_train[ind, 0, :])
        loss = torch.mean((y_h - Y_train[ind])**2)
        loss.backward()
        optimizer.step()
    torch.save(ndpn.state_dict(), model_save_path + '/model.pt')
    if epoch % 20 == 0:
        x_test = X_test[np.arange(100)]
        y_test = Y_test[np.arange(100)]
        y_htest = ndpn(x_test, y_test[:, 0, :])
        for j in range(18):
            plt.figure(figsize=(8, 8))
            plt.plot(0.667*y_h[j, :, 0].detach().cpu().numpy(), -0.667*y_h[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
            plt.axis('off')
            plt.savefig(model_save_path + '/train_img_' + str(j) + '.png')

            plt.figure(figsize=(8, 8))
            img = X_train[ind][j].cpu().numpy()*255
            img = np.asarray(img*255, dtype=np.uint8)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(model_save_path + '/ground_train_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

            plt.figure(figsize=(8, 8))
            plt.plot(0.667*y_htest[j, :, 0].detach().cpu().numpy(), -0.667*y_htest[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
            plt.axis('off')
            plt.savefig(model_save_path + '/test_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

            plt.figure(figsize=(8, 8))
            img = X_test[j].cpu().numpy()*255
            img = np.asarray(img*255, dtype=np.uint8)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(model_save_path + '/ground_test_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
        test = ((y_htest - y_test)**2).mean(1).mean(1)
        print('Epoch: ' + str(epoch) + ', Test Error: ' + str(test.mean().item()))
