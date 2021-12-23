import argparse
import os
from utils import utils
import torch
import data

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--data_name', type=str, default="avazu", help="name of dataset: avazu|criteo")
        parser.add_argument('--name', type=str, default="temp", help="name of training")
        parser.add_argument('--checkpoints_dir', type=str, default="../logs", help="checkpoint director")
        parser.add_argument('--pre_dir', type=str, default="../pre_model", help="pre-trained models director")
        # parser.add_argument('--log_dir', type=str, default="../logs", help="log director")
        parser.add_argument('--print_freq', type=int, default=500, help="every batch num for loss output")
        parser.add_argument('--valid_freq', type=int, default=3000, help="every batch num for valid metrics output")
        parser.add_argument('--pre_train_quat', action="store_true", help="load pre model with the trained quantization model")
        parser.add_argument('--pre_train', action="store_true", help="load pre model")
        parser.add_argument("--pre_quat", action="store_true", help="load pre quantization model")
        parser.add_argument('--dim', type=int, default=32, help="dimension of fm model")
        parser.add_argument('--weight_decay', type=float, default=1e-6, help="weight decay for optimizer")
        parser.add_argument('--batch_size', type=int, default=20480, help="trainng batchsize")
        parser.add_argument('--valid_batch_size', type=int, default=20480, help="trainng batchsize")
        parser.add_argument('--test_batch_size', type=int, default=20480, help="trainng batchsize")
        parser.add_argument('--arch_learning', type=float, default=0.001, help = "learning rate for optimizer")
        parser.add_argument('--learning', type=float, default=0.0001, help = "learning rate for optimizer")
        parser.add_argument("--epoch", type=int, default=30, help="training epoch num")
        parser.add_argument("--data_size", type=int, default=-1, help="size for training data")
        parser.add_argument("--K", type=int, default=512, help="size of codebook")
        parser.add_argument("--M", type=int, default=4, help="num of codebook")
        parser.add_argument("--threshold", type=int, default=150, help="field dimension threshold for pq")
        parser.add_argument("--test_data_size", type=int, default=-1, help="size for test data")
        parser.add_argument("--times", type=int, default=1, help="the num of run in this program")
        parser.add_argument("--optimizer", type= str, default="adam", help="type of optimizer")
        parser.add_argument("--loss", type= str, default="bce", help="type of loss function")
        parser.add_argument("--model", type=str, default="fm", help="name of model")
        parser.add_argument("--gpu_ids", type=str, default="1")    
        parser.add_argument("--unrolled", action="store_true", help="the type for nas algo")    
        parser.add_argument("--record", type=int, default=0, help="whether save model")
        parser.add_argument("--share", type=int, default=0, help="share codebook for different size or not")    
        parser.add_argument("--dis_loss", type=str, default="none", help="type of distance loss: none|avg|avg_import|weigth|weight_import")
        parser.add_argument("--save_quat", action="store_true", help="whther save quatization model")
        parser.add_argument("--retrain", action="store_true", help="retrain for new arch")
        parser.add_argument("--fix_arch", action="store_true", help="")
        parser.add_argument("--hardsoft", action="store_false", help="")
        parser.add_argument("--frequence", type=int, default=10, help="frequence of arch update")
        parser.add_argument("--division", type=int, default=10, help="frequence of arch update")
        parser.add_argument("--bucket_mode", type=str, default="random", help="frequence of arch update")
        parser.add_argument("--memory_limit", type=int, default=-1, help="the constraint for memory comsuption")
        parser.add_argument("--limit_method", type=int, default=0, help="type of limited arch update method")
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # save and return the parser
        opt, _ = parser.parse_known_args()
        dataset_name = opt.data_name
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.data_name, opt.model, opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'train_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt