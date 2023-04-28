import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import option
from model import Model
from dataset import Dataset
from utils import save_best_record, load_pre_model
import os
def cal_false_alarm(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    # false_num=0.
    # _len=len(labels)
    # for score,label in zip(scores,labels):
    #     if label!=score:
    #         false_num+=1
    fp=np.sum(scores*(1-labels))
    return fp/np.sum(1-labels)
def test(dataloader, model, args, device):

    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        count = 0
        if args.dataset == 'sh':
            gt = np.load('../list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('../list/gt-ucf.npy')
        elif args.dataset == 'xd':
            gt = np.load('../list/gt-xd.npy')
        # print(gt.shape)
        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, logits, feat, _, _ = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
            np_pred = sig.squeeze(-1).cpu().detach().numpy()
            np_pred = np.repeat(np_pred, 16)
            # np_pred = (np_pred - np.min(np_pred)) / (np.max(np_pred) - np.min(np_pred))
            temp_gt = gt[count:count+np_pred.shape[0]]


            res = np.array([np_pred,temp_gt])

            count += np_pred.shape[0]
        
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)

        roc_auc = auc(fpr, tpr)
        print('roc_auc : ' + str(roc_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall     , precision)
        print('pr_auc : ' + str(pr_auc))

        torch.cuda.empty_cache()

        return roc_auc


if __name__ == '__main__':
 
    args = option.parser.parse_args()
    # args.dataset = 'sh'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(args.feature_size, args.batch_size, 12, 4,3, device)
    model = load_pre_model(args.pretrained_ckpt, model)
    model = model.to(device)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=8, pin_memory=False)
    test(test_loader, model, args, device)
