import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
import matplotlib.pyplot as plt
def cos(a,b):
    res = np.sum(a*b.T)/(np.sqrt(np.sum(a * a.T)) * np.sqrt(np.sum(b * b.T)))
    return res
def cosScoreAndClipValue(net1, net2):
    '''net1 - gobal, net2 - local'''
    nparr1 = np.array([])
    nparr2 = np.array([])
    for key, var in net1.items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr1 = np.append(nparr1, nplist)
    for key, var in net2.items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr2 = np.append(nparr2, nplist)
    return cos(nparr1, nparr2), norm_clip(nparr1, nparr2)
def norm_clip(nparr1, nparr2):
    '''v - nparr1, v_clipped - nparr2'''
    vnum = np.linalg.norm(nparr1, ord=None, axis=None, keepdims=False)
    return vnum / np.linalg.norm(nparr2, ord=None, axis=None, keepdims=False)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=10, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=100, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=2000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    acc_list=[]
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        FLTrustTotalScore = 0
        FLTrustCentralNorm = myClients.centralTrain(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            #计算cos相似度得分和向量长度裁剪值
            client_score, client_clipped_value = cosScoreAndClipValue(FLTrustCentralNorm, local_parameters)
            FLTrustTotalScore += client_score
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    #乘得分 再乘裁剪值
                    sum_parameters[key] = client_score * client_clipped_value * var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + client_score * client_clipped_value * local_parameters[var]

        for var in global_parameters:
            #除以所以客户端的得分总和
            global_parameters[var] = (sum_parameters[var] / FLTrustTotalScore)

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))
                acc_list.append(sum_accu.item() / num)
                print(acc_list)

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
    print(acc_list)
    plt.plot(acc_list)
    plt.show()
