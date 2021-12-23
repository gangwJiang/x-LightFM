import os
import time


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Visualizer():

    def __init__(self, opt):
        self.opt = opt  # cache the option

        mkdir(os.path.join(opt.checkpoints_dir, opt.data_name, opt.model, opt.name))
        self.train_log_name = os.path.join(opt.checkpoints_dir, opt.data_name, opt.model, opt.name, 'train_log.txt')
        with open(self.train_log_name, "w") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        self.log_name = os.path.join(opt.checkpoints_dir, opt.data_name, opt.model, opt.name, 'valid_log.txt')
        with open(self.log_name, "w") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Valid Metrics (%s) ================\n' % now)

    def print_valid_losses(self, epoch, losses, t_comp, iters=-1):
        if iters == -1:
            message = '(epoch: %d, time: %.3f) ' % (epoch, t_comp)
        else:
            message = '(epoch: %d, iter: %d, time: %.3f) ' % (epoch, iters, t_comp)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)
        print("Valid: ", message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_losses(self, epoch, iters, task, losses, t_comp):
        message = '(epoch: %d, process: %d/%d, time: %.3f) ' % (epoch, iters, task, t_comp)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print("Train: ", message)  # print the message
        with open(self.train_log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
    
    def retrain_begin(self):
        with open(self.train_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Retrain Training Loss (%s) ================\n' % now)
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Retrain Valid Metrics (%s) ================\n' % now)