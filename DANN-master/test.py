import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
import torchvision
from torchvision import datasets
import random
import GPUtil
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from model import CNNModel
import argparse

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

    parser.add_argument('--load_model', type=str, default='best_model')
    # model save name

    parser.add_argument('--batch_size', type=int, default=1)
    # batch size

    args, _ = parser.parse_known_args()
    return args

args = parse_args()

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

print(torch.cuda.is_available())


model_root = "{}.pth".format(args.load_model)

cuda = True
cudnn.benchmark = True
batch_size = 1
alpha = 0

"""load data"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
])

# dataset 및 dataloader
test_dir = "data/ADNI2_test"
test_dataset = torchvision.datasets.ImageFolder(root=test_dir,transform=transform)

test_loader_source = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False)

""" training """

my_net = CNNModel()
my_net.load_state_dict(torch.load(os.path.join(model_root)))
my_net.eval()
print('Load model : {}'.format(model_root))

if cuda:
    my_net = my_net.cuda()

len_dataloader = len(test_loader_source)
data_target_iter = iter(test_loader_source)

i = 0
n_total = 0
n_correct = 0
y_true = []
y_pred = []

while i < len_dataloader:

    # test model using target data
    data_target = next(data_target_iter)
    t_img, t_label = data_target

    batch_size = len(t_label)

    if cuda:
        t_img = t_img.cuda()
        t_label = t_label.cuda()

    class_output, _ = my_net(input_data=t_img, alpha=alpha)
    _, pred = torch.max(class_output, 1)

    y_pred.append(pred.cpu().detach().numpy())
    y_true.append(t_label.cpu().detach().numpy())

    i += 1

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
accuracy = (tp+tn) / (tp+tn+fp+fn)
sensitivity = tp / (tp+fn)
specificity = tn / (tn+fp)

print('###########################################\n')
print('Test : {}'.format(model_root))
print('Accuracy(ACC) on the target domain : ', accuracy)
print('Sensitivity(SEN) on the target domain : ', sensitivity)
print('Specificity(SPE) on the target domain : ', specificity)
print('AUC curve(AUC) on the target domain : ', roc_auc_score(y_true, y_pred))
print('\n###########################################')