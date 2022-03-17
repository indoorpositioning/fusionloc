import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from DataSource import *

EPOCHS = 500
BATCH_SIZE = 64

class PoseNet(torch.nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        # resnet model to be used as encoder, with a 2048
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        print(self.resnet)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # Parameters of newly constructed modules have requires_grad=True by default
        self.fc_features = 2048
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, self.fc_features)
        self.fc_pose_xyz = torch.nn.Linear(self.fc_features, 3)
        self.fc_pose_wpqr = torch.nn.Linear(self.fc_features, 4)

    def forward(self, x):
        out = self.resnet(x)
        out = F.dropout(out, training=self.training)
        out_xyz = self.fc_pose_xyz(out)
        out_wpqr = self.fc_pose_wpqr(out)
        out_pose = torch.cat((out_xyz, out_wpqr), dim=1)
        return out_pose
\
# Basic Loss FUnction
class PoseNetCriterion(torch.nn.Module):
    def __init__(self, beta = 512.0):
        super(PoseNetCriterion, self).__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.beta = beta

    def forward(self, y, t):
        # Translation loss
        loss = self.loss_fn(y[:, :3], t[:, :3])
        # Rotation loss
        ori_out = F.normalize(y[:, 3:], p=2, dim=1)
        ori_true = F.normalize(t[:, 3:], p=2, dim=1)
        loss += self.beta * self.loss_fn(ori_out, ori_true)
        return loss

# Baysian PoseNet Loss Function
class PoseLoss(torch.nn.Module):
    def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -6.25
            
        self.sx = torch.nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = torch.nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

        if learn_beta:
            self.sx.requires_grad = True
            self.sq.requires_grad = True

        self.loss_print = None

    def forward(self, pred, target):
        pred_q =  pred[:, 3:]
        pred_x = pred[:, :3]
        target_q =  target[:, 3:]
        target_x = target[:, :3]

        pred_q = F.normalize(pred_q, p=2, dim=1)
        target_q = F.normalize(target_q, p=2, dim=1)
        loss_x = F.l1_loss(pred_x, target_x)
        loss_q = F.l1_loss(pred_q, target_q)

            
        loss = torch.exp(-self.sx)*loss_x \
               + self.sx \
               + torch.exp(-self.sq)*loss_q \
               + self.sq

        return loss


if __name__ == "__main__":

    # Set up Data Loaders
    # training data
    data_path = './'
    train_dataset = DataSource(data_path, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # validation data
    val_dataset = DataSource(data_path, train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = PoseNet()
    model = model.to(device)

    criterion = PoseLoss(device, 0, -6.25, True)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                {'params': [criterion.sx, criterion.sq]}], lr=1e-4, weight_decay=0.0005)#, weight_decay=0.0001)

    # track learning curve
    train_iters, train_losses = [], []
    val_iters, val_losses = [], []
    # training
    n_train, n_val = 0, 0 # the number of iterations (for plotting)

    # Uncomment if want to restore/use checkpoint and train
    # check_path = './check_epoch_3499_nograd_beta250'
    # checkpoint = torch.load(check_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # ep = checkpoint['epoch']
    # loss = checkpoint['loss']

    print("begin training\n")

    #comment if want to use checkpoint
    ep = 0
    # Begin training
    for epoch in range(ep, EPOCHS):
        print("Epoch: " + str(epoch) + "\n")
        model.train()
        for step, (imgs, poses) in enumerate(train_loader):
            imgs = imgs.to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            poses = torch.Tensor(poses).to(device)

            out = model(imgs)
            loss = criterion(out, poses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # save the current training information
            train_iters.append(n_train)
            train_losses.append(float(loss)/BATCH_SIZE)   # compute *average* loss
            n_train += 1
            if step % 5 == 0:
                print(" iteration:" + str(step) + "\n    " + " Training Loss is: " + str(loss))

        # validate model every 5 epochs and checkpoint mode
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                for step, (imgs, poses) in enumerate(val_loader):
                    imgs = imgs.to(device)
                    poses[0] = np.array(poses[0])
                    poses[1] = np.array(poses[1])
                    poses[2] = np.array(poses[2])
                    poses[3] = np.array(poses[3])
                    poses[4] = np.array(poses[4])
                    poses[5] = np.array(poses[5])
                    poses[6] = np.array(poses[6])
                    poses = np.transpose(poses)
                    poses = torch.Tensor(poses).to(device)

                    out = model(imgs)
                    loss = criterion(out, poses)

                    print(" iteration: " + str(step) + "\n   " + " Validation Loss is: " + str(loss))

                     # save the current validation information
                    val_iters.append(n_val)
                    val_losses.append(float(loss)/BATCH_SIZE)   # compute *average* loss
                    n_val += 1
            model.train()        
        if epoch % 5 == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "check_epoch_{}".format(epoch))

    #Save final model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "check_epoch_{}".format(epoch))

    # plotting
    plt.title("Learning Curve")
    plt.plot(train_iters, train_losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig('curve.png')





            

