import torch
from torch import nn
import torch.nn.functional as F

class PointNetLoss(nn.Module):
    def __init__(self, device, w=0.0001):
        super(PointNetLoss, self).__init__()
        self.w = w
        self.nll_loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, gt, pr, A_):
        A = A_.clone()
        #  Orthogonality constraint
        orth = torch.norm(torch.eye(A.shape[1]).to(self.device) - torch.matmul(A, A.transpose(1, 2)))
        loss = self.nll_loss(pr, gt) + self.w * orth
        return loss

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
        
        #self.sx = self.sx.to(device)
        #self.sq = self.sq.to(device)

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

        #self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss

class AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sx=0.0, sq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        loss = torch.exp(-self.sx) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sx + \
               torch.exp(-self.sq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.sq
        return loss

if __name__ == "__main__":
    batch_size = 5
    classes = 15

    pred = torch.randn(batch_size, classes, requires_grad=True)
    target = torch.empty(batch_size, dtype=torch.long).random_(classes)
    A = torch.rand(batch_size, 64, 64)

    print("pred.shape: ",pred.shape, "target.shape: ",target.shape, "A.shape", A.shape)
    # pred.shape:  torch.Size([5, 15]) 
    # target.shape:  torch.Size([5]) 
    # A.shape torch.Size([5, 64, 64])


    loss = PointNetLoss()
    output = loss(target, pred, A)

    print(output)
