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
from loss import MLSLoss, KLDiracVMF
from UACL.Uncer import UncertaintyHead, UncertaintyHead_MLS

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


# Training function
def _train(model, train_loader, optimizer, epoch, loss_fn, scheduler, uncer):
    train_losses_tem = []
    num = 0
    for pair in train_loader:
        num += 1
        x, y = pair[0], pair[1]
        x = x.cuda().float().contiguous()
        y = y.cuda().long().contiguous()  # [B]
        optimizer.zero_grad()

        feature, sig_feat = model(x, mode='embedding')

        log_sig_sq = uncer(sig_feat)  # (N, 1) log
        loss = loss_fn(feature, log_sig_sq, y)

        loss.backward()
        loss = loss.item()
        optimizer.step()
        train_losses_tem.append(loss)
    # scheduler.step()
    return sum(train_losses_tem) / len(train_losses_tem)


# Emotional recognition training function
def _train_epochs(model, train_loader, epochs, lr, uncer):
    # optimizer = optim.Adam(uncer.parameters(), lr=lr, weight_decay=5e-4)
    optimizer = optim.Adam(uncer.parameters(), lr=lr)
    loss_fn = MLSLoss(mean=False)
    # loss_fn = KLDiracVMF(z_dim=512, radius=64)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    # 开始训练
    min_train_loss = 1000
    for epoch in range(1, epochs + 1):
        model.eval()
        uncer.train()
        # model.train()
        train_losses_tem = _train(model, train_loader, optimizer, epoch, loss_fn, scheduler, uncer)
        if train_losses_tem <= min_train_loss:
            min_train_loss = train_losses_tem
            saved_models_dir = 'Pretrained_MLS'
            torch.save(uncer,
                       os.path.join(saved_models_dir, saved_models_dir + '_best.pth'))
        print(f'Epoch {epoch}, Train loss {train_losses_tem:.4f}')

    return train_losses


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

epochs = 2000
# lr = 3e-5
lr = 1e-4
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = MLSLoss(mean=True)
# loss_fn = KLDiracVMF(z_dim=512, radius=64)
# Import pre-trained model
ssl_root = './Pretrained_Arcface_SphereFace/Pretrained_SphereFace_best.pth'
model = ResNet()
model = torch.load(ssl_root)
model.to(device)
uncer = UncertaintyHead_MLS()
uncer.to(device)

# Training
train_losses = _train_epochs(model, train_loader, epochs, lr, uncer)
