import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import pandas as pd
def cal_false_alarm(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    # false_num=0.
    # _len=len(labels)
    # for score,label in zip(scores,labels):
    #     if label!=score:
    #         false_num+=1
    fp=np.sum(scores*(1-labels))
    return fp/np.sum(1-labels)
def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def test(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, logits,_,_,_ = model(inputs=input)

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            sig = logits
            pred = torch.cat((pred, sig))
        # print('pred:',pred.shape)
        if args.dataset == 'sh':
            gt = np.load('../list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('../list/gt-ucf.npy')
        elif args.dataset == 'xd':
            gt = np.load('../list/gt-xd.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)

        roc_auc = auc(fpr, tpr)
        print('roc_auc : ' + str(roc_auc))


        # best_threshold=threshold[np.where((tpr-fpr)==np.max(tpr-fpr))]

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))

        torch.cuda.empty_cache()

        return roc_auc,pr_auc

