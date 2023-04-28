import torch
import torch.nn as nn
from torch.autograd.function import Function as F
import torch.optim as optim


def softmaxA(x, omega=1, dim=-1):
    x_exp_1 = (x * omega).exp() - 1  # m * n
    partition = x_exp_1.sum(dim=dim, keepdim=True) + 1e-10  # 按列累加, m * 1
    return x_exp_1 / partition

def cosdis(x1,x2):
    return (1-torch.cosine_similarity(x1,x2,dim=-1))/2

class CenterLoss(nn.Module):
    def __init__(self, feat_dim, device=torch.device('cuda'), alpha=0.001, beta=0.5,th=0.9, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(feat_dim, device=device))

        # self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.th = float(th)
        self.alpha = alpha
        self.beta = beta
        self.norm = nn.LayerNorm(feat_dim)
        self.tripletloss = nn.TripletMarginWithDistanceLoss(margin=1.0,distance_function=cosdis)
    def forward(self, feat, score):
        assert feat.size(-1) == self.feat_dim
        nor_feat = feat[:int(feat.size(0) // 2)]
        abn_faet = feat[int(feat.size(0) // 2):]
        abn_score = score[int(feat.size(0) // 2):].squeeze(-1)

        nor_feat = nor_feat.view(-1, self.feat_dim)

        center_batch_size = nor_feat.size(0)

        loss = 1-(torch.cosine_similarity(nor_feat,self.centers,dim=-1).mean())

        centers = self.centers.clone().detach()
        abn_index = torch.topk(abn_score, 6, largest=True,dim=-1)[1].unsqueeze(-1)
        nor_index = torch.topk(abn_score, 6, largest=False, dim=-1)[1].unsqueeze(-1)
        # print(abn_index[-1],nor_index[-1])
        AAF = torch.gather(abn_faet, 1, abn_index.repeat(1, 1, self.feat_dim))
        ANF = torch.gather(abn_faet, 1, nor_index.repeat(1, 1, self.feat_dim))

        loss2 = self.tripletloss(centers,ANF,AAF)
        # loss2 = max(triplet_1 - triplet_2 + 1, 0)
        print('center_loss',loss, loss2)
        loss_total = loss * self.alpha+ loss2*self.beta

        return loss_total


class CenterlossFunc(F):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


def main(test_cuda=False):
    print('-' * 80)

    device = torch.device("cuda" if test_cuda else "cpu")
    ct = CenterLoss(2, size_average=True).to(device)
    optimizer = optim.Adam(ct.parameters(),
                           lr=0.1, weight_decay=0.005)
    y = torch.Tensor([[0.5, 1], [0.99, 0.1], [0.5, 1], [0.99, 0.1]]).to(device)
    feat = torch.ones(4, 2).to(device)
    print(list(ct.parameters()))
    for i in range(50):
        print('centers.grad:', ct.centers.grad)
        out = ct(feat, y)
        print('out', out.item())
        optimizer.zero_grad()
        out.backward()
        optimizer.step()
        # ct.centers-=ct.centers.grad
        print(ct.centers.grad)
        print('center:', ct.centers)
        # print(feat.grad)


if __name__ == '__main__':
    torch.manual_seed(999)
    main(test_cuda=False)
    # if torch.cuda.is_available():
    #     main(test_cuda=True)
