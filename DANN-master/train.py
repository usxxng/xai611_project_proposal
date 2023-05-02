import random
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision import datasets
import torchvision
from torchvision import transforms
from tqdm import tqdm
from model import CNNModel
import numpy as np
import GPUtil
import warnings
import argparse

warnings.filterwarnings("ignore")

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
random.seed(random_seed)
np.random.seed(random_seed)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="0")
    # gpu id number

    parser.add_argument('--model_name', type=str, default='best_model')
    # model save name

    parser.add_argument('--batch_size', type=int, default=16)
    # batch size

    parser.add_argument('--source', type=str, default='adni1')
    # source domain

    parser.add_argument('--target', type=str, default='adni2')
    # target domain

    parser.add_argument('--init_lr', type=float, default=1e-4)
    # learning rate

    parser.add_argument('--epochs', type=int, default=100)

    args, _ = parser.parse_known_args()
    return args

args = parse_args()

source_dataset_name = 'adni1'
target_dataset_name = 'adni2'

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

print(torch.cuda.is_available())

cuda = True
cudnn.benchmark = True

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
])

# dataset 및 dataloader
source_dir = "data/ADNI1"
source_dataset = torchvision.datasets.ImageFolder(root=source_dir, transform=transform)

target_dir = "data/ADNI2"
target_dataset = torchvision.datasets.ImageFolder(root=target_dir, transform=transform)

class_names = source_dataset.classes

dataset_sizes = {}
dataset_sizes["train"] = int(0.8 * len(source_dataset))
dataset_sizes["val"] = len(source_dataset) - dataset_sizes["train"]

datasets = {}
datasets["train"], datasets["val"] = torch.utils.data.random_split(source_dataset, [dataset_sizes["train"], dataset_sizes["val"]])

train_loader_source = torch.utils.data.DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
valid_loader_source = torch.utils.data.DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)

train_loader_target = torch.utils.data.DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
valid_loader_target = torch.utils.data.DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=args.init_lr)
scheduler = CosineAnnealingWarmRestarts(optimizer, args.epochs - 1)

loss_function = nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    criterion = loss_function.cuda()

for p in my_net.parameters():
    p.requires_grad = True

def do_epoch(my_net, data_loader_source, data_loader_target, criterion, optim=None):
    train_loss = 0
    train_acc = 0
    total_size = 0
    y_true = []
    y_pred = []
    len_dataloader = min(len(data_loader_source), len(data_loader_target))
    data_source_iter = iter(data_loader_source)
    data_target_iter = iter(data_loader_target)

    for i in tqdm(range(len_dataloader), leave=False):
        p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        data_source = next(data_source_iter)
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        domain_label = torch.zeros(batch_size).long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()

        class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
        err_s_label = criterion(class_output, s_label)
        err_s_domain = criterion(domain_output, domain_label)

        _, preds = torch.max(class_output, 1)
        total_size += s_label.size(0)

        # training model using target data
        data_target = next(data_target_iter)
        t_img, _ = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        _, domain_output = my_net(input_data=t_img, alpha=alpha)
        err_t_domain = criterion(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        if optim is not None:
            optim.zero_grad()
            err.backward()
            optimizer.step()

        train_acc += (preds == s_label).sum().item()
        train_loss += err.item()

        y_pred.append(preds.cpu().detach().tolist())
        y_true.append(s_label.cpu().detach().tolist())

    auc_score = roc_auc_score(sum(y_true, []), sum(y_pred, []), average="macro")
    train_loss = train_loss / len_dataloader
    train_acc = train_acc / total_size

    return train_loss, train_acc, auc_score

# training
best_loss = 100000
best_auc = 0
for epoch in range(args.epochs):
    my_net.train()
    train_loss, train_acc, train_auc = do_epoch(my_net, train_loader_source, train_loader_target, loss_function, optim=optimizer)

    my_net.eval()
    with torch.no_grad():
        val_loss, val_acc, val_auc = do_epoch(my_net, valid_loader_source, valid_loader_target, loss_function, optim=None)

    tqdm.write(f'Epoch {epoch:03d}: train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, train_auc = {train_auc:.4f} ----- valid_loss = {val_loss:.4f}, valid_acc = {val_acc:.4f}, valid_auc = {val_auc:.4f}')

    if val_auc > best_auc:
        tqdm.write(f'Saving model... Selection: val_auc')
        best_auc = val_auc
        torch.save(my_net.state_dict(), "{}.pth".format(args.model_name))

    scheduler.step()

print('done')
