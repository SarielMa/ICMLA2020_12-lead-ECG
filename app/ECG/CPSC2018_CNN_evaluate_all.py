# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:22:11 2020

@author: linhai
"""
import sys
import os
sys.path.append('../../core')
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import torchvision.models as tv_models
from Evaluate import cal_performance, cal_AUC_robustness
from Evaluate_mask import test_rand, test_adv
#from Evaluate_advertorch import test_adv
#from Evaluate_bba_spsa import test_adv as test_adv_spsa
from CPSC2018_Dataset import get_dataloader #, get_dataloader_bba
import csv
#%%
class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                               dilation=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(out_planes, out_planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(out_planes, out_planes)
        self.downsample = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.gn3 = nn.GroupNorm(out_planes, out_planes)
    def forward(self, x):

        #print(x.shape)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        #print(out.shape)

        out = self.conv2(out)
        out = self.gn2(out)

        #print(out.shape)

        x = self.downsample(x)
        x = self.gn3(x)

        #print(x.shape)

        out += x
        out = self.relu(out)

        return out

class Resnet18(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer0 = nn.Sequential(nn.Conv1d(12, 64, kernel_size=11, stride=2,
                                              dilation=1, padding=5, bias=False),
                                    nn.GroupNorm(64, 64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.layer1 = nn.Sequential(BasicBlock(64, 64, 2),  BasicBlock(64, 64, 2))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128, 2))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256, 2))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512, 2),
                                    nn.Conv1d(512, 512, 3, 1, 0),
                                    nn.ReLU(inplace=True))
        #self.layer5 = nn.Sequential(nn.AdaptiveAvgPool1d(1),
        #                            nn.Flatten(),
        #                            nn.GroupNorm(1, 512, affine=False),
        #                            nn.Linear(512, 9))
        self.layer5=nn.Linear(512, 9)
        self.mask_avgpool=nn.AvgPool1d(kernel_size=3072, stride= 1024, padding=0)

        self.tau=0.5
        
        self.pool='max'

    def forward(self, x, mask):
        #print(x.shape)
        x = self.layer0(x)
        #print('0', x.shape)
        x = self.layer1(x)
        #print('1', x.shape)
        x = self.layer2(x)
        #print('2', x.shape)
        x = self.layer3(x)
        #print('3', x.shape)
        x = self.layer4(x)
        #print('4', x.shape)

        if self.pool == 'avg':
            m=self.mask_avgpool(mask)
            #m.shape: [batch_size, 1, 31]
            m[m<0.5]=0
            #m[m>=0.5]=1
            x=x*m
            #x.shape: [batch_size, 512, 31]
            #avg pooling
            x=torch.sum(x, dim=2) #x.shape: [batch_size, 512]
            m=torch.sum(m, dim=2) #m.shape  [batch_size, 512]
            x=x/(m+1e-8)

            x = self.layer5(x)
            #x.shape: [batch_size, 9]

            #x=x/(self.tau*torch.sqrt(torch.mean(x**2, dim=1, keepdim=True))+1e-8)
            
        elif self.pool == 'max':
            #mask x is unnecessary
            #m=self.mask_avgpool(mask)
            #m.shape: [batch_size, 1, 31]
            #m[m<0.5]=0
            #m[m>=0.5]=1
            #x=x*m
            # x.shape: [batch_size, 512, 31]
            x=x.permute(0, 2, 1)
            #x.shape: [batch_size, 31, 512]
            x=self.layer5(x) 
            #x.shape: [batch_size, 31, 9]
            x=torch.max(x, dim=1)[0]            
            #x.shape: [batch_size, 9]
        return x
#%%
class Block(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64,
                               kernel_size=15, stride=1, padding=7, bias=bias)
        self.bn1 = nn.GroupNorm(4, 64)
        self.dp1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64,
                               kernel_size=16, stride=2, padding=7, bias=bias)
        self.bn2 = nn.GroupNorm(4,64)
        self.dp2 = nn.Dropout(0.2)
        self.pool = nn.MaxPool1d(kernel_size=16, stride=2,padding = 7)
        #self.pool = nn.MaxPool1d(kernel_size=5, stride=1)
        #self.linear1 = Linear(17997*32, 64*12, bias=bias)
        #self.linear2 = Linear(64*12, 9*12, bias=bias)
    def forward(self, x):
        x0 = self.pool(x)
        x1=self.conv1(self.dp1(nnF.relu(self.bn1(x), inplace=True)))
        x2=self.conv2(self.dp2(nnF.relu(self.bn2(x1), inplace=True)))
        x3 = x2+x0
        return x3

class Net1(nn.Module):
    def __init__(self,bias=True):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=16, stride=2,padding = 7)
        self.conv0 = nn.Conv1d(in_channels=12, out_channels=64,
                               kernel_size=15, stride=1, padding=7, bias=bias)
        self.bn0 = nn.GroupNorm(4,64)

        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64,
                               kernel_size=15, stride=1, padding=7, bias=bias)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64,
                               kernel_size=16, stride=2, padding=7, bias=bias)
        self.bn1 = nn.GroupNorm(4,64)
        self.dp1 = nn.Dropout(0.2)
        block_list = []
        for i in range(14):
            block_list.append(Block(bias))
        self.block = nn.ModuleList(block_list)

        self.bn2 = nn.GroupNorm(4,64)
        self.linear1 = nn.Linear(64*2, 9, bias=bias)
        #self.linear2 = nn.Linear(24, 9, bias=bias)

    def forward(self, x):
        #input size (N,C,L), C is 12,L is 72000
        #X = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]
        #Z0 = []
        #for i, x in enumerate(X):
        #X=X.view(X.size(0),1,X.size(1))
        x = nnF.relu(self.bn0(self.conv0(x)))

        x0= self.pool(x)

        x1=self.dp1(nnF.relu(self.bn1(self.conv1(x)), inplace=True))
        x1 = self.conv2(x1)
        x1 = x1+x0


        for i in range(14):
            x1 = self.block[i](x1)
        x3 = x1

        x3 = nnF.relu(self.bn2(x3), inplace = True)
        x3 = x3.view(x3.size(0),-1)
        z=self.linear1(x3)
        #z=self.linear2(x4)
        #z=nnF.softmax(z, dim=0)
        return z
#%%
def Net(net_name):
    if net_name == 'resnet18a':
        model=Resnet18()
        model.pool='avg'
        return model
    elif net_name == 'resnet18m':
        model=Resnet18()
        model.pool='max'
        return model
    elif net_name =='Net1':
        return Net1()
#%%
def save_checkpoint(filename, model, result, epoch):
    state_dict = model.state_dict()
    torch.save({'epoch': epoch,
                'model_state_dict': state_dict,
                'result':result},
               filename)
    print('saved:', filename)
#%%
def test(model, device, dataloader, num_classes, class_balanced_acc=False):
    model.eval()#set model to evaluation mode
    sample_count=0
    sample_idx_wrong=[]
    confusion=np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            X, Y = batch_data[0].to(device), batch_data[1].to(device)
            Mask = batch_data[2].to(device)
            Z = model(X, Mask)
            Yp = Z.data.max(dim=1)[1] #multiclass/softmax
            for i in range(0, num_classes):
                for j in range(0, num_classes):
                    confusion[i,j]+=torch.sum((Y==i)&(Yp==j)).item()
            #------------------
            for n in range(0,X.size(0)):
                if Y[n] != Yp[n]:
                    sample_idx_wrong.append(sample_count+n)
            sample_count+=X.size(0)
    #------------------
    acc, sens, prec = cal_performance(confusion, class_balanced_acc)
    result={}
    result['confusion']=confusion
    result['acc']=acc
    result['sens']=sens
    result['prec']=prec
    result['sample_idx_wrong']=sample_idx_wrong
    print('testing')
    print('acc', result['acc'])
    print('sens', result['sens'])
    print('prec', result['prec'])
    return result
#%%
def plot_result(loss_train_list, acc_train_list,
                acc_val_list, acc_test_list):
    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].set_title('loss v.s. epoch')
    ax[0].plot(loss_train_list, '-b', label='training loss')
    ax[0].set_xlabel('epoch')
    #ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title('accuracy v.s. epoch')
    ax[1].plot(acc_train_list, '-b', label='train acc')
    ax[1].plot(acc_val_list, '-r', label='val acc')
    ax[1].set_xlabel('epoch')
    #ax[1].legend()
    ax[1].grid(True)
    ax[2].set_title('accuracy v.s. epoch')
    ax[2].plot(acc_test_list, '-m', label='test acc')
    ax[2].set_xlabel('epoch')
    #ax[2].legend()
    ax[2].grid(True)
    return fig, ax
#%%
def get_filename(net_name, loss_name, epoch=None, pre_fix='resultA/CPS2018_CNN_'):
    if epoch is None:
        filename=pre_fix+net_name+'_'+loss_name
    else:
        filename=pre_fix+net_name+'_'+loss_name+'_epoch'+str(epoch)
    return filename
#%%
def main(epoch_start, epoch_end, train, arg, evaluate_model):
    main_train(epoch_start, epoch_end, train, arg)
    if evaluate_model == True:
        main_evaluate(epoch_end-1, arg)
#%%
def main_evaluate(epoch, arg):
    device=arg['device']
    norm_type=arg['norm_type']
    net_name=arg['net_name']
    loss_name=arg['loss_name']
    #loader_bba = get_dataloader_bba()
    loader_train, loader_val, loader_test = get_dataloader(batch_size=32)
    del loader_train, loader_val
    #main_evaluate_rand(net_name, loss_name, epoch, device, loader_test, (0.05, 0.1, 0.2, 0.3, 0.4, 0.5))
    if norm_type == np.inf:
        noise_norm_list=(0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.1)
        #noise_norm_list=(0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10)
        #noise_norm_list=(0.001, 0.005, 0.01, 0.05, 0.10)
        print('Linf norm noise_norm_list', noise_norm_list)
    else:
        noise_norm_list=(1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
        print('L2 norm noise_norm_list', noise_norm_list)
    #main_evaluate_bba_spsa(net_name, loss_name, epoch, device, loader_bba, norm_type, noise_norm_list)
    #main_evaluate_wba(net_name, loss_name, epoch, device, 'bba', loader_bba, norm_type, noise_norm_list)
    main_evaluate_wba(net_name, loss_name, epoch, device, 'test', loader_test, norm_type, noise_norm_list)
#%%
def main_train(epoch_start, epoch_end, train, arg):
#%%
    net_name=arg['net_name']
    loss_name=arg['loss_name']
    filename=get_filename(net_name, loss_name)
    print('train model: '+filename)
    if epoch_start == epoch_end:
        print('epoch_end is epoch_start, exist main_train')
        return
    #---------------------------------------
    device=arg['device']
    lr=arg['lr']
    norm_type=arg['norm_type']
    rand_pad=arg['rand_pad']
#%%
    num_classes=9
    if norm_type == np.inf:
        noise_norm=0.1
        max_iter=1
        step=1.0
    elif norm_type == 2:
        noise_norm=5.0
        max_iter=1
        step=1.0
#%%
    loader_train, loader_val, loader_test = get_dataloader(rand_pad=rand_pad)
#%%
    loss_train_list=[]
    acc_train_list=[]
    acc_val_list=[]
    acc_test_list=[]
    epoch_save=epoch_start-1
#%%
    model=Net(net_name)
    if epoch_start > 0:
        print('load', filename+'_epoch'+str(epoch_save)+'.pt')
        checkpoint=torch.load(filename+'_epoch'+str(epoch_save)+'.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        #------------------------
        loss_train_list=checkpoint['result']['loss_train_list']
        acc_train_list=checkpoint['result']['acc_train_list']
        acc_val_list=checkpoint['result']['acc_val_list']
        acc_test_list=checkpoint['result']['acc_test_list']
        if 'E' in arg.keys():
            if arg['E'] is None:
                arg['E']=checkpoint['result']['arg']['E']
                print('load E')
    #------------------------
    model.to(device)
    #------------------------
    if arg['optimizer']=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif arg['optimizer']=='AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif arg['optimizer']=='Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    elif arg['optimizer']=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001, nesterov=True)
    else:
        raise NotImplementedError('unknown optimizer')
#%%
    for epoch in range(epoch_save+1, epoch_end):
        #-------- training --------------------------------
        start = time.time()
        loss_train, acc_train =train(model, device, optimizer, loader_train, epoch, arg)
        loss_train_list.append(loss_train)
        acc_train_list.append(acc_train)
        print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
        end = time.time()
        print('time cost:', end - start)
        #-------- validation --------------------------------
        result_val = test(model, device, loader_val, num_classes=num_classes, class_balanced_acc=True)
        acc_val_list.append(result_val['acc'])
        #-------- test --------------------------------
        result_test = test(model, device, loader_test, num_classes=num_classes, class_balanced_acc=True)
        acc_test_list.append(result_test['acc'])
        #--------save model-------------------------
        result={}
        result['arg']=arg
        result['loss_train_list'] =loss_train_list
        result['acc_train_list'] =acc_train_list
        result['acc_val_list'] =acc_val_list
        result['acc_test_list'] =acc_test_list
        if (epoch+1)%10 == 0:
            save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, result, epoch)
        epoch_save=epoch
        #------- show result ----------------------
        #plt.close('all')
        display.clear_output(wait=False)
        fig, ax = plot_result(loss_train_list, acc_train_list, acc_val_list, acc_test_list)
        display.display(fig)
        fig.savefig(filename+'_epoch'+str(epoch)+'.png')
        plt.close(fig)
#%%
def main_evaluate_wba(net_name, loss_name, epoch, device, loader, norm_type, noise_norm_list):
#%%
    filename=get_filename(net_name, loss_name, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))
    x=loader.dataset[0][0]
    model=Net(net_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(x.dtype).to(device)
    model.eval()
    print('evaluate_wba model in '+filename+'.pt')
    print(noise_norm_list)
#%% 100pgd
    num_repeats=1
    result_100pgd=[]
    for noise_norm in noise_norm_list:
        result_100pgd.append(test_adv(model, device, loader, num_classes=9,
                                      noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/10, method='pgd',
                                      clip_X_min=-1, clip_X_max=1,
                                      num_repeats=num_repeats, class_balanced_acc=True))
    noise=[0]
    acc=[result_100pgd[0]['acc_clean']]
    for k in range(0, len(result_100pgd)):
        noise.append(result_100pgd[k]['noise_norm'])
        acc.append(result_100pgd[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    result_100pgd[0]['auc']=auc
    #fig, ax = plt.subplots()
    #ax.plot(noise, acc, '.-b')
    #ax.set_ylim(0, 1)
    #ax.set_yticks(np.arange(0, 1.05, step=0.05))
    #ax.grid(True)
    #title='wba_100pgd_L'+str(norm_type)+' r'+str(num_repeats)+' auc='+str(auc)+' '+data_name
    #ax.set_title(title)
    #ax.set_xlabel(filename)
    #display.display(fig)
    #fig.savefig(filename+'_'+title+'.png')
    #plt.close(fig)
    #filename=filename+'_result_wba_L'+str(norm_type)+'_'+data_name+'.pt'
    #torch.save({'result_100pgd':result_100pgd},
    #           filename)
    #print('saved:', filename)
    return acc

#%% add rand noise to image
def main_evaluate_rand(net_name, loss_name, epoch, device, loader, noise_norm_list):
#%%
    filename=get_filename(net_name, loss_name, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))
    x=loader.dataset[0][0]
    model=Net(net_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(x.dtype).to(device)
    model.eval()
    print('evaluate_rand model in '+filename+'.pt')
    result_rand=[]
    for noise_norm in noise_norm_list:
        result_rand.append(test_rand(model, device, loader, num_classes=9, noise_norm=noise_norm,
                                     clip_X_min=-1, clip_X_max=1))
    noise=[0]
    acc=[result_rand[0]['acc_clean']]
    adv=[0]
    for k in range(0, len(result_rand)):
        noise.append(result_rand[k]['noise_norm'])
        acc.append(result_rand[k]['acc_noisy'])
        adv.append(result_rand[k]['adv_sample_count']/result_rand[k]['sample_count'])
    auc=cal_AUC_robustness(acc, noise)
    #result_rand[0]['auc']=auc
    #fig, ax = plt.subplots(1,2)
    #ax[0].plot(noise, acc, '.-b')
    #ax[0].set_ylim(0, 1)
    #ax[0].set_yticks(np.arange(0, 1.05, step=0.05))
    #ax[0].grid(True)
    #ax[0].set_title('rand')
    #ax[0].set_xlabel(filename)
    #ax[1].plot(noise, adv, '.-b')
    #ax[1].grid(True)
    #ax[1].set_title('rand adv%'+' auc='+str(auc))
    #display.display(fig)
    #fig.savefig(filename+'_rand.png')
    #plt.close(fig)
    #------------------------------------
    #filename=filename+'_result_rand.pt'
    #torch.save({'result_rand':result_rand}, filename)
    #print('saved:', filename)
    return acc




def show_ACC_wba():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='result_B/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("ACC")
    ax.set_xlim(0.1)
    noise_norm_list=(0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.1)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Loss2" in fn and "wba" in fn and "10adv" in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            checkpoint=torch.load(filename, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_100pgd']
            acc = [acc_all[0]['acc_clean']]
            print (fn)
            for line in acc_all:
                acc.append(line['acc_noisy'])

            if "1.0Loss2" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "1.0NSR")  
                accs.append(acc)
                list_labels.append("1.0NSR")
        elif "advls" in fn and "wba" in fn:
            print (fn)
            checkpoint=torch.load(pre_fix+fn, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_100pgd']
            acc = [acc_all[0]['acc_clean']]
            for line in acc_all:
                acc.append(line['acc_noisy'])
            if "0.005Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.005")
                accs.append(acc)
                list_labels.append("advls_0.005")
            if "0.01Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.01")
                accs.append(acc)
                list_labels.append("advls_0.01")
            if "0.05Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.05")
                accs.append(acc)
                list_labels.append("advls_0.05")
            if "0.1Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.1")
                accs.append(acc)
                list_labels.append("advls_0.1")
        elif "CPS2018_CNN_resnet18a_ce_Adam_rpT_epoch69" in fn and "wba" in fn:
            print (fn)
            checkpoint=torch.load(pre_fix+fn, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_100pgd']
            acc = [acc_all[0]['acc_clean']]
            for line in acc_all:
                acc.append(line['acc_noisy'])

            col += 1
            ax.plot(acc, colors[col],label = "CE")
            accs.append(acc)
            list_labels.append("CE")
        if "Jacob" in fn and "wba" in fn :
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            checkpoint=torch.load(filename, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_100pgd']
            acc = [acc_all[0]['acc_clean']]
            print (fn)
            for line in acc_all:
                acc.append(line['acc_noisy'])
            if "54.0Jacob" in fn:
                #col += 1
                ax.plot(acc, colors[col],label = "54.0Jacob",marker = "o")
                accs.append(acc)
                list_labels.append("54.0Jacob")
            if "44.0Jacob" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "44.0Jacob")      
                accs.append(acc)
                list_labels.append("44.0Jacob")            
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    fig.savefig(pre_fix+'Evaluate_wba_ACC.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)  
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="wha_test", flag="acc" )  

def F1(confusion):
    num_classes = 9
    F1s = np.zeros(num_classes)
    for n in range(0, num_classes):
        F1s[n] = (confusion[n,n]*2)/(np.sum(confusion[n,:])+np.sum(confusion[:,n]))
    return np.mean(F1s)

def show_F1_wba():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='result_B/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("F1")
    ax.set_xlim(0.1)
    noise_norm_list=(0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.1)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Loss2" in fn and "wba" in fn and "10adv" in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            checkpoint=torch.load(filename, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_100pgd']
            clean_confusion_matrix = acc_all[0]["confusion_clean"]
            acc = [F1(clean_confusion_matrix)]
            print (fn)
            for line in acc_all:
                acc.append(F1(line['confusion_noisy']))

            if "1.0Loss2" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "1.0NSR") 
                accs.append(acc)
                list_labels.append("1.0NSR")
        elif "advls" in fn and "wba" in fn:
            checkpoint=torch.load(pre_fix+fn, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_100pgd']
            clean_confusion_matrix = acc_all[0]["confusion_clean"]
            acc = [F1(clean_confusion_matrix)]
            print (fn)
            for line in acc_all:
                acc.append(F1(line['confusion_noisy']))
            if "0.005Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.005")
                accs.append(acc)
                list_labels.append("advls_0.005")
            if "0.01Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.01")
                accs.append(acc)
                list_labels.append("advls_0.01")
            if "0.05Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.05")
                accs.append(acc)
                list_labels.append("advls_0.05")
            if "0.1Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.1")
                accs.append(acc)
                list_labels.append("advls_0.1")
        elif "CPS2018_CNN_resnet18a_ce_Adam_rpT_epoch69" in fn and "wba" in fn:
            checkpoint=torch.load(pre_fix+fn, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_100pgd']
            clean_confusion_matrix = acc_all[0]["confusion_clean"]
            acc = [F1(clean_confusion_matrix)]
            print ("clean CE acc is ...",acc)
            print (fn)
            for line in acc_all:
                acc.append(F1(line['confusion_noisy']))
            col += 1
            ax.plot(acc, colors[col],label = "CE")
            accs.append(acc)
            list_labels.append("CE")
        elif "Jacob" in fn and "wba" in fn :
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            checkpoint=torch.load(filename, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_100pgd']
            clean_confusion_matrix = acc_all[0]["confusion_clean"]
            acc = [F1(clean_confusion_matrix)]
            print (fn)
            for line in acc_all:
                acc.append(F1(line['confusion_noisy']))
            if "54.0Jacob" in fn:
                #col += 1
                ax.plot(acc, colors[col],label = "54.0Jacob",marker = "o")
                accs.append(acc)
                list_labels.append("54.0Jacob")
            if "44.0Jacob" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "44.0Jacob")
                accs.append(acc)
                list_labels.append("44.0Jacob")
            
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    fig.savefig(pre_fix+'Evaluate_wba_F1.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)     
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="wha_test", flag="F1" )       

def show_ACC_rand():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='result_B/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("ACC")
    noise_norm_list=(0.05, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Loss2" in fn and "rand" in fn and "10adv" in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            checkpoint=torch.load(filename, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_rand']
            acc = [acc_all[0]['acc_clean']]
            print (fn)
            for line in acc_all:
                acc.append(line['acc_noisy'])
            """    
            if "0.9Loss2" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "0.9NSR")
            """
            if "1.0Loss2" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "1.0NSR") 
                accs.append(acc)
                list_labels.append("1.0NSR")
        elif "advls" in fn and "rand" in fn:
            print (fn)
            checkpoint=torch.load(pre_fix+fn, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_rand']
            acc = [acc_all[0]['acc_clean']]
            for line in acc_all:
                acc.append(line['acc_noisy'])
            if "0.005Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.005")
                accs.append(acc)
                list_labels.append("advls_0.005")
            if "0.01Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.01")
                accs.append(acc)
                list_labels.append("advls_0.01")
            if "0.05Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.05")
                accs.append(acc)
                list_labels.append("advls_0.05")
            if "0.1Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.1")
                accs.append(acc)
                list_labels.append("advls_0.1")
                
        elif "CPS2018_CNN_resnet18a_ce_Adam_rpT_epoch69" in fn and "rand" in fn:
            print (fn)
            checkpoint=torch.load(pre_fix+fn, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_rand']
            acc = [acc_all[0]['acc_clean']]
            for line in acc_all:
                acc.append(line['acc_noisy'])

            col += 1
            ax.plot(acc, colors[col],label = "CE")
            accs.append(acc)
            list_labels.append("CE")
        elif "Jacob" in fn and "rand" in fn :
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            checkpoint=torch.load(filename, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_rand']
            acc = [acc_all[0]['acc_clean']]
            print (fn)
            for line in acc_all:
                acc.append(F1(line['confusion_noisy']))
            if "54.0Jacob" in fn:
                #col += 1
                ax.plot(acc, colors[col],label = "54.0Jacob",marker = "o")
                accs.append(acc)
                list_labels.append("54.0Jacob")
            if "44.0Jacob" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "44.0Jacob")   
                accs.append(acc)
                list_labels.append("44.0Jacob")
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    fig.savefig(pre_fix+'Evaluate_rand_ACC.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)     
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="rand_test", flag="acc" )    
    
def show_F1_rand():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='result_B/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("F1")
    noise_norm_list=(0.05, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Loss2" in fn and "rand" in fn and "10adv" in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            checkpoint=torch.load(filename, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_rand']
            clean_confusion_matrix = acc_all[0]["confusion_clean"]
            acc = [F1(clean_confusion_matrix)]
            print (fn)
            for line in acc_all:
                acc.append(F1(line['confusion_noisy']))
            """    
            if "0.9Loss2" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "0.9NSR")
            """
            if "1.0Loss2" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "1.0NSR")   
                accs.append(acc)
                list_labels.append("1.0NSR")
        elif "advls" in fn and "rand" in fn:
            print (fn)
            checkpoint=torch.load(pre_fix+fn, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_rand']
            clean_confusion_matrix = acc_all[0]["confusion_clean"]
            acc = [F1(clean_confusion_matrix)]
            for line in acc_all:
                acc.append(F1(line['confusion_noisy']))
            if "0.005Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.005")
                accs.append(acc)
                list_labels.append("advls_0.005")
            if "0.01Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.01")
                accs.append(acc)
                list_labels.append("advls_0.01")
            if "0.05Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.05")
                accs.append(acc)
                list_labels.append("advls_0.05")
            if "0.1Linf" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "advls_0.1")
                accs.append(acc)
                list_labels.append("advls_0.1") 
                
        elif "CPS2018_CNN_resnet18a_ce_Adam_rpT_epoch69" in fn and "rand" in fn:
            print (fn)
            checkpoint=torch.load(pre_fix+fn, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_rand']
            acc = [acc_all[0]['acc_clean']]
            for line in acc_all:
                acc.append(line['acc_noisy'])

            col += 1
            ax.plot(acc, colors[col],label = "CE")
            accs.append(acc)
            list_labels.append("CE")
        elif "Jacob" in fn and "rand" in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            checkpoint=torch.load(filename, map_location=torch.device('cpu'))
            acc_all = checkpoint['result_rand']
            clean_confusion_matrix = acc_all[0]["confusion_clean"]
            acc = [F1(clean_confusion_matrix)]
            print (fn)
            for line in acc_all:
                acc.append(F1(line['confusion_noisy']))
                
            if "54.0Jacob" in fn:
                #col += 1
                ax.plot(acc, colors[col],label = "54.0Jacob",marker = "o")
                accs.append(acc)
                list_labels.append("54.0Jacob")           
            if "44.0Jacob" in fn:
                col += 1
                ax.plot(acc, colors[col],label = "44.0Jacob")  
                accs.append(acc)
                list_labels.append("44.0Jacob")
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    fig.savefig(pre_fix+'Evaluate_rand_F1.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="rand_test", flag="F1" )
    
def show_ACC_loss2():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    #plt.figure(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='result_Loss2/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("ACC")
    ax.set_xlim(0.1)
    noise_norm_list=(0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.1)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Loss2" in fn and "wba" in fn and "10adv" in fn and "val" in fn and "png" not in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------

            

            for beta in [0.4,0.6,0.8,0.9,1.0,1.1,1.2]:
                lt = str(beta)+"Loss2"
                if lt in fn:
                    print (fn)
                    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
                    acc_all = checkpoint['result_100pgd']
                    acc = [acc_all[0]['acc_clean']]
                    for line in acc_all:
                        acc.append(line['acc_noisy'])
                    else:
                        col += 1
                        ax.plot(acc, colors[col],label = str(beta)+"NSR")
                    accs.append(acc)
                    list_labels.append(str(beta)+"NSR")
                  
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    
    fig.savefig(pre_fix+'Evaluate_loss2_ACC.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="NSR_validation", flag="acc" )

def show_F1_loss2():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    #plt.figure(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='result_Loss2/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("F1")
    ax.set_xlim(0.1)
    noise_norm_list=(0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.1)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Loss2" in fn and "wba" in fn and "10adv" in fn and "val" in fn and "png" not in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            for beta in [0.4,0.6,0.8,0.9,1.0,1.1,1.2]:
                lt = str(beta)+"Loss2"
                if lt in fn:
                    print (fn)
                    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
                    acc_all = checkpoint['result_100pgd']
                    clean_confusion_matrix = acc_all[0]["confusion_clean"]
                    acc = [F1(clean_confusion_matrix)]
                    print (fn)
                    for line in acc_all:
                        acc.append(F1(line['confusion_noisy']))
                    #if beta == 1.01:
                    #    ax.plot(acc, colors[col],label = lt, marker="o")
                    else:
                        col += 1
                        ax.plot(acc, colors[col],label =  str(beta)+"NSR")
                    accs.append(acc)
                    list_labels.append(str(beta)+"NSR")
                  
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    
    fig.savefig(pre_fix+'Evaluate_loss2_F1.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="NSR_validation", flag="F1" )
    
def show_ACC_Jacob():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    #plt.figure(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='result_Jacob/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("ACC")
    ax.set_xlim(0.1)
    noise_norm_list=(0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.1)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Jacob" in fn and "wba" in fn  and "val" in fn and "png" not in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------

            

            for beta in [4.0,14.0,24.0,34.0,44.0,54.0,64.0,74.0,84.0]:
                lt = "resnet18a"+str(beta)+"Jacob"
                if lt in fn:
                    col += 1
                    print (fn)
                    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
                    acc_all = checkpoint['result_100pgd']
                    acc = [acc_all[0]['acc_clean']]
                    for line in acc_all:
                        acc.append(line['acc_noisy'])
                        
                    if col>6:
                        ax.plot(acc, colors[col%6],label = str(beta)+"Jacob", marker="o")
                    else:
                        
                        ax.plot(acc, colors[col],label = str(beta)+"Jacob")
                    accs.append(acc)
                    list_labels.append(str(beta)+"Jacob")
                  
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    
    fig.savefig(pre_fix+'Evaluate_Jacob_ACC.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="Jacob_validation", flag="acc" )


def show_F1_Jacob():
    print ("show F1 jacob")
    # for adv and
    #NUM_COLORS = 20
    fig, ax = plt.subplots(figsize=(6,8))
    #plt.figure(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='result_Jacob/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("F1")
    ax.set_xlim(0.1)
    noise_norm_list=(0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.1)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Jacob" in fn and "wba" in fn  and "val" in fn and "png" not in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------
            for beta in [4.0,14.0,24.0,34.0,44.0,54.0,64.0,74.0,84.0]:
                lt = "resnet18a"+str(beta)+"Jacob"
                #acc = []
                if lt in fn:
                    col +=1
                    print (fn)
                    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
                    acc_all = checkpoint['result_100pgd']
                    clean_confusion_matrix = acc_all[0]["confusion_clean"]
                    acc = [F1(clean_confusion_matrix)]
                    
                    for line in acc_all:
                        acc.append(F1(line['confusion_noisy']))
                    if col>6:

                        ax.plot(acc, colors[col%6],label = str(beta)+"Jacob", marker="o")
                        
                    else:
                        
                        ax.plot(acc, colors[col],label = str(beta)+"Jacob")
                    accs.append(acc)
                    list_labels.append(str(beta)+"Jacob")
                  
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    
    fig.savefig(pre_fix+'Evaluate_Jacob_F1.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="Jacob_validation", flag="F1" )
    
def get_table(accs,list_labels,noise_levels, pre_fix, dataset,  flag):
    #fig, ax = plt.subplots(1, 3)
    #colors = ['-b','-g','-r','-y','-k','-m','-c']
    #markers = ['.','s','^']
    outpath1 =pre_fix + "ACC_"+dataset
    outpath2 =pre_fix + "F1_"+dataset
    
    if flag == "acc":
        with open(outpath1+".csv",'w') as csvf:
            fw = csv.writer(csvf)
            fw.writerow([""]+noise_levels)
            for i,l in enumerate(list_labels):
                fw.writerow([l+"(ACC)"]+accs[i])
            #fw.writerow([l+"(spsa)"]+bba_spsa_accs[i])
    else:
        with open(outpath2+".csv",'w') as csvf:
            fw = csv.writer(csvf)
            fw.writerow([""]+noise_levels)
            for i,l in enumerate(list_labels):
                fw.writerow([l+"(F1)"]+accs[i])
if __name__ == '__main__':
    show_F1_wba()
    show_ACC_wba()
    show_F1_rand()
    show_ACC_rand()
    show_ACC_loss2()
    show_F1_loss2()
    show_F1_Jacob()
    show_ACC_Jacob()