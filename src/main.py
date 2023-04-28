import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record,load_pre_model
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from config import *
from center_loss import CenterLoss
import copy
from  torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR

def main(alpha, beta , depth,heads,LR , indx,topK):
    args = option.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(args.feature_size, args.batch_size, depth,heads,topK,device)
    if args.pretrained_ckpt is not None and args.pretrained_ckpt !='':
        model = load_pre_model(args.pretrained_ckpt,model)
    model_name = '{}i3d_{}_fc1fc21024_td{}h{}_C{}_{}_k{}'.format(args.dataset,LR,depth,heads,alpha,beta,topK)

    best_dict = model.state_dict()
    model = model.to(device)


    if not os.path.exists('../ckpt'):
        os.makedirs('../ckpt')
    
    
    loss_model = CenterLoss(feat_dim=1024, alpha=alpha, beta=beta, th=alpha, device=device)
    optimizer = optim.Adam(model.parameters(),
                           lr=LR, weight_decay=0.005)
    optimizer2 = optim.Adam(loss_model.parameters(),
                            lr=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=2, T_mult=2)

    test_info = {"epoch": [], "test_AUC": [],'ap':[]}
    best_ROC_AUC = -1
    best_AP_AUC = -1
    output_path = '../record'  # put your own path here
    roc_auc,ap = test(test_loader, model, args, device)
    # exit(0)
    file_path = os.path.join(output_path, model_name + '.txt')
    if os.path.exists(file_path):
        os.remove(file_path)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)


        train(loadern_iter, loadera_iter, model, loss_model, args.batch_size, optimizer, optimizer2, device,None)
        # scheduler.step(step/len(train_nloader))
        if step % 10 == 0 and step > 10:

            roc_auc,ap_auc = test(test_loader, model, args, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(roc_auc)
            test_info["ap"].append(ap_auc)
            if test_info["ap"][-1] > best_AP_AUC:
                best_ROC_AUC = test_info["test_AUC"][-1]
                best_AP_AUC = test_info["ap"][-1]
                best_dict = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
                save_best_record(test_info, file_path)
            # if step == 500 and best_ROC_AUC < 0.90:
            #     print("over!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #     break
        if (step - 1) % len(train_aloader) == 0:
            scheduler.step()
            scheduler2.step()
    os.rename(file_path, file_path.replace('.txt', '_' + str(round(best_ROC_AUC, 4))+'_'+str(round(best_AP_AUC, 4)) + '.txt'))
    torch.save(best_dict, '../ckpt/' + model_name + '_' + str(round(best_ROC_AUC, 4)) + '_'+str(round(best_AP_AUC, 4)) +'.pkl')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    count = 0
    for alpha in [0.001]:
        for beta in [0.5]:
            for depth in [6]:
                for heads in [4]:
                    for topK in [3]:
                        for LR in [1e-3]:
                            for repeat in range(3):
                                main(alpha, beta,depth,heads,LR, count,topK)
                                count += 1
