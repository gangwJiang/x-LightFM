from collections import Counter
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import os
import sys
sys.path.append("/amax/home/gangwei/project/LightFM")

from data import create_dataset
from model import create_model
from options.base_options import BaseOptions
import torch
import numpy as np
import time
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    opt = BaseOptions().parse()

    dataset, valid_dataset, test_dataset, field_dims = create_dataset(opt)
    print(field_dims)
    opt.field_dims = field_dims
    
    if "nfm" in opt.model:
        pth = "nfm_%d.pt" % (opt.dim)
    elif "deepfm" in opt.model:
        pth = "deepfm_%d.pt" % (opt.dim)
    elif "mdfm" in opt.model:
        pth = "mdfm_%d.pt" % (opt.dim)
    else:
        pth = "fm_%d.pt" % (opt.dim)
    
    if opt.pre_train_quat:
        opt.pre_model_path = os.path.join(opt.checkpoints_dir, opt.data_name, opt.model, opt.name, "model.pth")
    else:
        opt.pre_model_path = os.path.join(opt.pre_dir, opt.data_name, "models", pth)

    # opt.pre_model_path = "/amax/home/gangwei/project/dataset/fm_32.pt"
    offsets_a = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    model = create_model(opt).to(device)
    model.initial_freq_vector()
    print(model.frequence_vector)

    model.eval()
    field_list, targets, predicts = list(), list(), list()
    loss_sum = 0
    cnt = 0
    with torch.no_grad():
        for i, (fields, target) in enumerate(test_dataset):
            if opt.test_data_size < i*20480 and opt.test_data_size != -1:
                break
            fields = fields.to(device=device, dtype=torch.long)
            target = target.to(device)
            y, loss = model.test(fields, target)
            loss_sum += loss
            cnt += 1
            field_list.extend(fields.tolist())
            targets.extend(target.tolist())
            predicts.extend(y.tolist())


    well_targets, well_predicts = list(), list()
    bad_field_list, bad_targets, bad_predicts = list(), list(), list()
    well_field_list, well_predicts_z, bad_predicts_z = list(), list(), list()
    threshold = 0.9
    for i in range(len(targets)):
        z = 1/(1 + np.exp(-predicts[i]))
        if abs(targets[i]-z) > threshold:
            bad_targets.append(targets[i])
            bad_predicts.append(predicts[i])
            bad_field_list.append(field_list[i])
            # bad_predicts_z.append(z)
        elif abs(targets[i]-z) < 0.1:
            well_targets.append(targets[i])
            well_predicts.append(predicts[i])
            well_field_list.append(field_list[i])
            # well_predicts_z.append(z)
    print("total: ", len(targets),roc_auc_score(targets, predicts))
    print("well: ", len(well_targets),roc_auc_score(well_targets, well_predicts))
    print("bad: ", len(bad_targets), roc_auc_score(bad_targets, bad_predicts))
    # print("well: ", len(well_targets),roc_auc_score(well_targets, well_predicts_z))
    # print("bad: ", len(bad_targets), roc_auc_score(bad_targets, bad_predicts_z))
    print(bad_predicts[:50])
    # print(bad_predicts_z[:50])
    print(bad_targets[:50])
    print(well_predicts[:50])
    # print(well_predicts_z[:50])
    print(well_targets[:50])

    fv = torch.from_numpy(model.frequence_vector).to(device)
    x = torch.LongTensor(bad_field_list).to(device)
    x = x + x.new_tensor(offsets_a).unsqueeze(0)
    x_fv = F.embedding(x, fv)
    x_fv_np = x_fv.cpu().numpy()
    inds = [3,4,6,9,10,11,14]
    
    popular = np.zeros((20, x_fv.shape[1]))
    for j in range(x_fv.shape[0]):
        for k in range(x_fv.shape[1]):
            # print(x_fv_np[j,k,1])
            popular[int(x_fv_np[j,k,1]*19.9),k] += 1 
    book_porp = np.zeros((20, len(inds)*4))
    for j in range(x_fv.shape[0]):
        for k in range(len(inds)):
            for z in range(4):
                book_porp[int(x_fv_np[j,inds[k],z*2+3]*19.9), k*4+z] += 1

    plt.figure(figsize=(10,6))
    ax = sns.heatmap(np.log(popular+1), linewidths=0.3, annot=False)
    plt.savefig("./visualize/bad_popular_log.jpg")

    plt.figure(figsize=(10,6))
    ax = sns.heatmap(np.log(book_porp+1), linewidths=0.3)
    plt.savefig("./visualize/bad_porp_log.jpg")
    
    print("------- well feature --------")
    x = torch.LongTensor(well_field_list).to(device)
    x = x + x.new_tensor(offsets_a).unsqueeze(0)
    x_fv = F.embedding(x, fv)

    x_fv_np = x_fv.cpu().numpy()

    popular = np.zeros((20, x_fv.shape[1]))
    for j in range(x_fv.shape[0]):
        for k in range(x_fv.shape[1]):
            popular[int(x_fv_np[j,k,1]*19.9),k] += 1 
    book_porp = np.zeros((20, len(inds)*4))
    for j in range(x_fv.shape[0]):
        for k in range(len(inds)):
            for z in range(4):
                book_porp[int(x_fv_np[j,inds[k],z*2+3]*19.9), k*4+z] += 1
    # for j in range(3):
    #     print(j)
    #     for k in range(len(x_fv[j,])):
    #         if x_fv[j, k, 0 ] != 0:
    #             print(field_dims[k], x_fv[j,k].cpu().numpy()[1:])

    plt.figure(figsize=(10,6))
    ax = sns.heatmap(np.log(popular+1), linewidths=0.3, annot=False)
    plt.savefig("./visualize/well_popular_log.jpg")

    plt.figure(figsize=(10,6))
    ax = sns.heatmap(np.log(book_porp+1), linewidths=0.3)
    plt.savefig("./visualize/well_porp_log.jpg")
    