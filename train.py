import torch
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

import os
import sys

from utils.utils import EarlyStopper
from data import create_dataset
from model import create_model
from utils.visualizer import Visualizer
from options.base_options import BaseOptions

import json
import numpy as np
import time


class Train(object):
    def __init__(self, model, opt, visual):
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.model = model.to(self.device)

        self.step = 0
        self.temperature = 2.5 * np.exp(-0.036 * self.step)

        self.best_arch = None
        self.visual = visual
        self.epoch_iter = 0
        self.cost = [0 for _ in range(10)]
        
        model_path = os.path.join(opt.checkpoints_dir, opt.data_name, opt.model, opt.name, "model.pth")
        self.early_stopper = EarlyStopper(num_trials=5, save_path=model_path, record=opt.record)

    def train(self, epoch, data_loader, test_data_loader, valid_data_loader, visualizer):
        self.model.train()
        self.epoch_iter = 0
        begin = time.time()
        self.iter_begin = time.time()
        if self.model.search:
            self.model.initial_validset(valid_data_loader)
        data_size = len(data_loader)*self.opt.batch_size if self.opt.data_size==-1 else self.opt.data_size
        for i, (fields, target) in enumerate(data_loader):
            self.epoch_iter += len(target)
            if self.opt.data_size < self.epoch_iter and self.opt.data_size != -1:
                break

            # add update b here
            fields = fields.to(device=self.device, dtype=torch.long)
            target = target.to(self.device)

            self.model.optimize_parameters(fields, target)

            if (i + 1) % self.opt.print_freq == 0:
                losses = self.model.get_current_losses()
                invert_op = getattr(self.model, "genotype", None)
                if callable(invert_op):
                    if not self.model.fix_arch:
                        invert_op()
                self.visual.print_current_losses(epoch, self.epoch_iter/10000, data_size/10000, losses, time.time()-self.iter_begin)
                self.iter_begin = time.time()

            if (i + 1) % self.opt.valid_freq == 0:
                loss = self.test(test_data_loader)
                visualizer.print_valid_losses(epoch, loss, 0, self.epoch_iter/10000)
        loss = self.test(test_data_loader)
        end = time.time()
        visualizer.print_valid_losses(epoch, loss, end-begin)
        if not self.early_stopper.is_continuable(task.model, loss, task):
            print('validation: best auc: %f' % self.early_stopper.best_accuracy)
            return True
        return False

    def test(self, data_loader):
        self.model.eval()
        targets, predicts = list(), list()
        loss_sum = 0
        cnt = 0
        with torch.no_grad():
            for i, (fields, target) in enumerate(data_loader):
                if self.opt.test_data_size < i*20480 and self.opt.test_data_size != -1:
                    break
                fields = fields.to(device=self.device, dtype=torch.long)
                target = target.to(self.device)

                y, loss = self.model.test(fields, target)
                loss_sum += loss
                cnt += 1
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                # print(torch.sigmoid(y))
        auc = roc_auc_score(targets, predicts)
        # loss = self.model.get_current_losses()
        loss = {'auc': auc, 'logloss': loss_sum/cnt, "memory": self.model.memory_cost}
        # loss = {'auc': auc, 'logloss': loss_sum/cnt}
        return loss

    def retrain(self):
        self.opt.batch_size = 20480
        self.opt.learning = 0.0001
        new_model = create_model(self.opt)
        self.model = new_model.to(self.device)
        print(self.best_arch)
        self.model.fix_arch = True
        self.model.set_arch(self.best_arch)
        self.early_stopper.best_accuracy = 0
        return

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        pre_state_dict = torch.load(path)
        self.model.copy(pre_state_dict)
        self.model.eval()


if __name__ == "__main__":
    opt = BaseOptions().parse()

    dataset, valid_dataset, test_dataset, field_dims = create_dataset(opt)
    print(field_dims)
    opt.field_dims = field_dims
    visualizer = Visualizer(opt)
    
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

    results = []
    for i in range(opt.times):
        model = create_model(opt)
        task = Train(model, opt, visualizer)
        loss = task.test(test_dataset)
        visualizer.print_valid_losses(-1, loss, 0)
        for j in range(opt.epoch):
            # if opt.model == "qfm_comp" and j >= 3:
            #     task.model.quat_freeze()
            success = task.train(j, dataset, test_dataset, valid_dataset, visualizer)
            if success:
                break

        if opt.retrain:
            # task.best_arch = [[241, 1, 1], [8, 1, 1], [8, 1, 1], [3697, 1024, 2], [4614, 1024, 4], [25, 1, 1], [5481, 1024, 1], [329, 1, 1], [31, 1, 1], [381763, 64, 2], [1611748, 2048, 2], [6793, 1024, 4], [6, 1, 1], [5, 1, 1], [2509, 512, 2], [9, 1, 1], [10, 1, 1], [432, 1, 1], [5, 1, 1], [68, 1, 1], [169, 1, 1], [61, 1, 1]]
            # task.best_arch =  [[241, 64], [8, 1], [8, 1], [3697, 1024], [4614, 1024], [25, 1], [5481, 2048], [329, 128], [31, 1], [381763, 64], [1611748, 512], [6793, 2048], [6, 1], [5, 1], [2509, 256], [9, 1], [10, 1], [432, 128], [5, 1], [68, 1], [169, 64], [61, 1]]
            # task.best_arch = task.model.random_arch()
            # task.best_arch = [[241, 1, 1], [8, 1, 1], [8, 1, 1], [3697, 512, 4], [4614, 1024, 4], [25, 1, 1], [5481, 256, 4], [329, 128, 2], [31, 1, 1], [381763, 512, 4], [1611748, 2048, 2], [6793, 2048, 1], [6, 1, 1], [5, 1, 1], [2509, 512, 2], [9, 1, 1], [10, 1, 1], [432, 1, 1], [5, 1, 1], [68, 1, 1], [169, 1, 1], [61, 1, 1]]            
            task.retrain()
            dataset, valid_dataset, test_dataset, field_dims = create_dataset(task.opt)
            visualizer.retrain_begin()
            loss = task.test(test_dataset)
            visualizer.print_valid_losses(-1, loss, 0)
            for j in range(100):
                success = task.train(j, dataset, test_dataset, valid_dataset, visualizer)
                if success:
                    break
            
        results.append([i, task.early_stopper.best_accuracy, task.early_stopper.best_memory])
        if opt.retrain:
            results[i].append(task.best_arch)
    print(results)
    log_file = os.path.join(opt.checkpoints_dir, opt.data_name, "summary.txt")
    with open(log_file, "a") as f:
        f.write(opt.model)
        f.write("  ")
        f.write(opt.name)
        f.write(":  ")
        # r_json = json.dumps(results)
        f.write(str(results))
        f.write("\n")

