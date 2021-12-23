import importlib
import torch.utils.data

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset, valid_dataset, test_dataset, field_dims = data_loader.load_data()
    return dataset, valid_dataset, test_dataset, field_dims


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.data_name)
        self.dataset = dataset_class(opt.dataset_path, opt.cache_path)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        
        dataset_len = len(self.dataset)
        train_length = int(dataset_len * 0.8)
        valid_length = int(dataset_len * 0.1)
        test_length = dataset_len - train_length - valid_length
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, 
            (train_length, valid_length, test_length), 
            generator=torch.Generator().manual_seed(19))
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=32)
        self.valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=opt.valid_batch_size,
            shuffle=True,
            num_workers=32)
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.test_batch_size,
            num_workers=32)
        # self.dim_importance = md_solver(torch.Tensor(dataset.field_dims.astype(np.float32)), 0.15, d0=self.dim, round_dim=False)
        # self.dim_importance = (self.dim_importance - self.dim_importance.min())/(self.dim_importance.max() - self.dim_importance.min())
        

    def load_data(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader, self.dataset.field_dims

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data