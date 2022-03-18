# -*- coding = utf-8 -*-
import torch
import matplotlib.pyplot as plt
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
# dict = './checkpoints/mnist_2nn_num_comm19_E5_B10_lr0.01_num_clients100_cf0.1'
# model1 = torch.load(dict)
# model2 = torch.load('./checkpoints/mnist_2nn_num_comm299_E5_B10_lr0.01_num_clients100_cf0.1')
# print(norm_clip(model1,model2))
acc_list =[0.18149993896484376, 0.556499900817871, 0.6716998291015625, 0.7579998016357422, 0.8413002014160156, 0.8471998596191406, 0.9054999542236328, 0.8280001831054687, 0.8501999664306641, 0.9144998931884766, 0.8991999816894531, 0.9091998291015625, 0.8923001098632812, 0.8963998413085937, 0.9419001007080078, 0.9487999725341797, 0.9246000671386718, 0.9383002471923828, 0.9163002777099609, 0.9444000244140625, 0.9511000061035156, 0.95010009765625, 0.9483001708984375, 0.9571001434326172, 0.9567000579833984, 0.9513001251220703, 0.9598001098632812, 0.9613999176025391, 0.9667999267578125, 0.938499755859375, 0.9564000701904297, 0.9660997772216797]

plt.plot(acc_list)
plt.show()