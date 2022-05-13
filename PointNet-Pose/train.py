import numpy as np
#import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from torchvision import transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import SevenScenesPoints
from network_v2 import PointNetPose
from network import PointNetPose_v1
from final_network import RGBPointLoc
from loss import PoseLoss, AtLocCriterion

EPOCHS = 500
BATCH_SIZE = 64
CHECK_PATH = "./models/check_epoch_65.pt"

if __name__ == "__main__":

    # Set up Data Loaders
    # training data
    data_path = '../Datasets/chess'
    train_dataset = SevenScenesPoints(data_path, train=True, num_samples=1024)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # validation data
    val_dataset = SevenScenesPoints(data_path, train=False, num_samples=1024)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model =  PointNetPose_v1(device)#RGBPointLoc(device)##PointNetPose()
    model = model.to(device)

    #criterion = PoseNetCriterion(beta=220)
    criterion = PoseLoss(device, 0, -6.25, True)  # alt sq=-6.25 instead of -3.0
    #criterion = AtLocCriterion(sq=-3.0, learn_beta=True)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                {'params': [criterion.sx, criterion.sq]}], lr=1e-4)#, weight_decay=0.0005)#, weight_decay=0.0001)
    #optimizer = torch.optim.SGD(nn.ParameterList(posenet.parameters()), lr=learning_rate)

    # track learning curve
    train_iters, train_losses = [], []
    val_iters, val_losses = [], []
    # training
    n_train, n_val = 0, 0 # the number of iterations (for plotting)

    # Uncomment if want to restore/use checkpoint and train
    if CHECK_PATH:
        check_path = CHECK_PATH
        checkpoint = torch.load(check_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ep = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        ep = 0
    print("begin training\n")

    #comment if want to use checkpoint
    #ep = 0
    # Begin training
    for epoch in range(ep, EPOCHS):
        print("Epoch: " + str(epoch) + "\n")
        model.train()
        for step, (rgbs, points, poses) in enumerate(train_loader):
            points = points.to(device)#torch.cat((imgs, depth), dim=1)
            # rgbs = rgbs.to(device)
            # depth = depth.to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            poses = torch.Tensor(poses).to(device)

            out = model(points)#model(rgbs, points)
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

        # validate model every 200 epochs and checkpoint mode
        if epoch % 5 == 0 and epoch != 0:
            epoch_loss = 0
            iter_count = 0
            model.eval()
            with torch.no_grad():
                for step, (rgbs, points, poses) in enumerate(val_loader):
                    points = points.to(device)
                    # rgbs = rgbs.to(device)
                    # imgs = imgs.to(device)
                    # depth = depth.to(device)
                    poses[0] = np.array(poses[0])
                    poses[1] = np.array(poses[1])
                    poses[2] = np.array(poses[2])
                    poses[3] = np.array(poses[3])
                    poses[4] = np.array(poses[4])
                    poses[5] = np.array(poses[5])
                    poses[6] = np.array(poses[6])
                    poses = np.transpose(poses)
                    poses = torch.Tensor(poses).to(device)

                    out =  model(points)#model(rgbs, points)
                    loss = criterion(out, poses)

                    print(" iteration: " + str(step) + "\n   " + " Validation Loss is: " + str(loss))

                     # save the current validation information
                    # val_iters.append(n_val)
                    # val_losses.append(float(loss)/BATCH_SIZE)   # compute *average* loss
                    epoch_loss += loss
                    iter_count += 1
                    n_val += 1
            print(" Epoch: " + str(epoch) + " Average Validation Loss is: " + str(epoch_loss/iter_count))        
            if epoch % 5 == 0:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, "./models/check_epoch_{}.pt".format(epoch))
            model.train()        

    #Save final model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "./models/final_{}.pt".format(epoch))