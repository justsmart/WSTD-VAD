import numpy as np
import torch
import torch.nn.functional as F
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss



def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class loss_SCL_V(torch.nn.Module):
    # Implementation Reference: https://github.com/tianyu0207/RTFM
    def __init__(self, alpha, margin,device):
        super(loss_SCL_V, self).__init__()
        self.device = device
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal
        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.to(self.device)

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        # return loss_cls
        loss_a_nor = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))+torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)
        loss_v = torch.mean(loss_a_nor ** 2)
        loss_total = loss_cls + self.alpha * loss_v
        return loss_total


def train(nloader, aloader, model,loss_model, batch_size, optimizer,optimizer2, device,pre_model):
    with torch.set_grad_enabled(True):
        model.train()
        # pre_model.eval()
        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal,scores, X,two_scores,atten= model(input)  # b*32  x 2048
        # with torch.no_grad():
        #     _,_,_,_,_,_,_,pre_atten = pre_model(input)
        scores = scores.view(batch_size * 32 * 2, -1)

        scores = scores.squeeze()
        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_criterion = loss_SCL_V(0.0001, 100,device)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn) + loss_smooth + loss_sparse
        # cost2 = loss_model(X, two_scores, torch.cat((nlabel,alabel)))

        cost2 = loss_model(X,atten)
        print(cost.item(),cost2.item())
        cost = cost+cost2
        print("train_cost:",cost.item())
        optimizer2.zero_grad()
        optimizer.zero_grad()
        cost.backward()
        # cost2.backward()
        optimizer.step()
        optimizer2.step()
        torch.cuda.empty_cache()


