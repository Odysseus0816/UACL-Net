import numpy as np
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from UACL.model import UACL
import pandas as pd
import random
from UACL.Uncer import UncertaintyHead, UncertaintyHead_MLS
from UACL.FC import FC_Class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def init_seeds(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False


# Test function
def _eval(model, test_loader, loss_fn,  uncer, fc_model):
    model.eval()
    uncer.eval()
    fc_model.eval()

    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for pair in test_loader:
            x, y = pair[0], pair[1]
            x = x.cuda().float().contiguous()
            y = y.cuda().long().contiguous()
            # out = model(x, mode='classifier')
            feature, sig_feat = model(x, mode='embedding')

            log_sig_sq = uncer(sig_feat)  # (N, 1) log
            # loss = loss_fn(feature, log_sig_sq, y)
            out = fc_model(log_sig_sq)

            acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float() / y.shape[0]).item()
            loss = loss_fn(out, y)
            loss = loss.item()
            total_loss += loss * y.shape[0]
            total_acc += acc * y.shape[0]
            test_loss = total_loss / len(test_loader.dataset)
            test_acc = total_acc / len(test_loader.dataset)
    return test_loss, test_acc


# Training function
def _train(model, train_loader, optimizer, epoch, loss_fn, uncer, fc_model):
    train_losses_tem = []
    train_acces_tem = []
    num = 0
    for pair in train_loader:
        num += 1
        x, y = pair[0], pair[1]
        x = x.cuda().float().contiguous()
        y = y.cuda().long().contiguous()
        optimizer.zero_grad()

        feature, sig_feat = model(x, mode='embedding')

        log_sig_sq = uncer(sig_feat)  # (N, 1) log
        # loss = loss_fn(feature, log_sig_sq, y)
        out = fc_model(log_sig_sq)

        acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float() / y.shape[0]).item()
        loss = loss_fn(out, y)
        loss.backward()
        loss = loss.item()
        optimizer.step()
        train_losses_tem.append(loss)
        train_acces_tem.append(acc)
        # print(f'Epoch: {epoch}, batch: {num}, Train_loss: {loss:.4f}, Train_acc: {acc:.4f}')
    return sum(train_losses_tem) / len(train_losses_tem), sum(train_acces_tem) / len(train_acces_tem)


# Emotional recognition training function
def _train_epochs(model, train_loader, test_loader, epochs, lr, uncer, fc_model):
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': uncer.parameters(), 'lr': 1e-3, 'weight_decay': 5e-4},
        {'params': fc_model.parameters(), 'lr': 1e-4}
    ])
    # optimizer = optim.Adam(fc_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    train_acces = []
    # test_loss, test_acc = _eval(model, test_loader, loss_fn)
    test_losses = []
    test_acces = []
    # print(f'Epoch {0}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}')
    # 开始训练
    min_train_loss =1000
    for epoch in range(1, epochs + 1):
        model.train()
        uncer.train()
        fc_model.train()
        train_losses_tem, train_acces_tem = _train(model, train_loader, optimizer, epoch, loss_fn, uncer, fc_model)
        print(f'Epoch: {epoch}, Train_loss: {train_losses_tem:.4f}, Train_acc: {train_acces_tem:.4f}')
        train_acces.append(train_acces_tem)
        train_losses.append(train_losses_tem)
        test_loss, test_acc = _eval(model, test_loader, loss_fn,  uncer, fc_model)
        test_losses.append(test_loss)
        test_acces.append(test_acc)
        if test_acc <= min_train_loss:
            min_train_loss = test_acc
            saved_models_dir = 'classification_stage'
            torch.save(fc_model,
                       os.path.join(saved_models_dir, saved_models_dir + '_best.pth'))
        print(f'Epoch {epoch}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}')
    return train_losses, train_acces, test_losses, test_acces


init_seeds(42)
# save_path = './SEED/'
# # Import data and set hyper-parameters
# x_train_path = save_path+'x_train_SEED.npy'
# x_test_path = save_path+'x_test_SEED.npy'
# y_train_path = save_path+'y_train_SEED.npy'
# y_test_path = save_path+'y_test_SEED.npy'
save_path = 'G:/xuxu/Project-15/EEG/Self-supervised-main/DEAP/'
x_train_path = save_path + 'x_train_DEAP.npy'
x_test_path = save_path + 'x_test_DEAP.npy'
y_train_path = save_path + 'y_train_DEAP.npy'
y_test_path = save_path + 'y_test_DEAP.npy'

x_train = np.load(x_train_path)
x_test = np.load(x_test_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)

x_train = x_train.reshape(-1, 1, x_train.shape[-2], x_train.shape[-1])
x_test = x_test.reshape(-1, 1, x_test.shape[-2], x_test.shape[-1])
y_train = y_train.reshape(-1, )
y_test = y_test.reshape(-1, )

batch_size = 512
train_dataset = data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

epochs = 1000
lr = 1e-2
loss_fn = nn.CrossEntropyLoss()

# Import pre-trained model
model_root = './Pretrained_Arcface_MLS_testing/Pretrained_SphereFace_best.pth'
model = UACL()
model = torch.load(model_root)
model.to(device)

uncer_root = './Pretrained_MLS/Pretrained_MLS_best.pth'
uncer = UncertaintyHead_MLS()
uncer = torch.load(uncer_root)
uncer.to(device)


fc_model = FC_Class()
fc_model.to(device)

# Training
_, _, _, acces = _train_epochs(model, train_loader, test_loader, epochs, lr, uncer, fc_model)

