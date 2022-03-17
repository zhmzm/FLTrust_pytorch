# -*- coding = utf-8 -*-
import torch
import numpy as np
def cos(a,b):
    res = np.sum(a*b.T)/(np.sqrt(np.sum(a * a.T)) * np.sqrt(np.sum(b * b.T)))
    return res

def norm_clip(v, v_clipped):
    nparr1 = np.array([])
    nparr2 = np.array([])
    for key, var in v.state_dict().items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr1 = np.append(nparr1, nplist)
    for key, var in v_clipped.state_dict().items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr2 = np.append(nparr2, nplist)
    vnum = np.linalg.norm(nparr1, ord=None, axis=None, keepdims=False)
    return vnum / np.linalg.norm(nparr2, ord=None, axis=None, keepdims=False)

def cosScore(net1, net2):
    nparr1 = np.array([])
    nparr2 = np.array([])
    for key, var in net1.state_dict().items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr1 = np.append(nparr1, nplist)
    for key, var in net2.state_dict().items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr2 = np.append(nparr2, nplist)

    return cos(nparr1, nparr2)
# tor_arr=torch.from_numpy(np_arr)
# tor2numpy=tor_arr.numpy()
dict = './checkpoints/mnist_2nn_num_comm19_E5_B10_lr0.01_num_clients100_cf0.1'
model1 = torch.load(dict)
model2 = torch.load('./checkpoints/mnist_2nn_num_comm299_E5_B10_lr0.01_num_clients100_cf0.1')
print(norm_clip(model1,model2))