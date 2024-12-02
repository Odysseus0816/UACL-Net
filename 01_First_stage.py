import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from UACL.model import UACL
import random
from loss import MLSLoss, KLDiracVMF, SphereFace

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
def _eval(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for pair in test_loader:
            x, y = pair[0], pair[1]
            x = x.cuda().float().contiguous()
            y = y.cuda().long().contiguous()
            out = model(x, mode='classifier')
            acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float() / y.shape[0]).item()
            loss = loss_fn(out, y)
            loss = loss.item()
            total_loss += loss * y.shape[0]
            total_acc += acc * y.shape[0]
            test_loss = total_loss / len(test_loader.dataset)
            test_acc = total_acc / len(test_loader.dataset)
    return test_loss, test_acc


# Training function
def _train(model, train_loader, optimizer, epoch, loss_fn, scheduler, arcface):
    train_losses_tem = []
    train_acc_tem = []
    num = 0
    for pair in train_loader:
        num += 1
        x, y = pair[0], pair[1]
        x = x.cuda().float().contiguous()
        y = y.cuda().long().contiguous()  # [B]
        optimizer.zero_grad()
        feature = model(x, mode='contrast')  # [B, 512]
        output = arcface(feature)
        arc_loss = loss_fn(output, y)  # ArcFace loss
        loss = arc_loss
        loss.backward()
        loss = loss.item()
        optimizer.step()
        """acc_arc：arcface输出的分类正确率；	"""
        acc_arc = torch.sum(torch.argmax(output, dim=1) == y) / batch_size
        train_losses_tem.append(loss)
        train_acc_tem.append(acc_arc)
    # scheduler.step()
    return sum(train_losses_tem) / len(train_losses_tem), sum(train_acc_tem) / len(train_acc_tem)


# Emotional recognition training function
def _train_epochs(model, train_loader, epochs, lr, arcface):
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': arcface.parameters()}
    ], lr=lr)
    # loss_fn = MLSLoss(mean=True)
    # loss_fn = KLDiracVMF(z_dim=512, radius=64)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    # 开始训练
    min_train_loss = 1000
    for epoch in range(1, epochs + 1):
        model.train()
        arcface.train()
        train_losses_tem, train_acc_tem = _train(model, train_loader, optimizer, epoch, loss_fn, scheduler, arcface)

        if train_losses_tem <= min_train_loss:
            min_train_loss = train_losses_tem
            saved_models_dir = 'Pretrained_SphereFace'
            torch.save(model,
                       os.path.join(saved_models_dir, saved_models_dir + '_best.pth'))
        print(f'Epoch {epoch}, Train loss {train_losses_tem:.4f}, , Test acc {train_acc_tem:.4f}')

    return train_losses


init_seeds(42)

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

epochs = 100
lr = 1e-3
arcface = SphereFace(feature_num=512, class_num=3)
arcface.to(device)
model = UACL()
model.to(device)

# Training
train_losses = _train_epochs(model, train_loader, epochs, lr, arcface)
