import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from collections import defaultdict

class MyTabularDataset(Dataset):

    def __init__(self, dataset, label_col):
        """
        :param dataset: dataset, DataFrame
        :param label_col: name of your column
        """

        self.label = torch.LongTensor(dataset[label_col].values)

        self.data = dataset.drop(columns=[label_col]).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label = self.label[index]
        data = self.data[index]

        return torch.tensor(data).float(), label

class MyImageDataset(Dataset):
    def __init__(self, dataset, file_col, label_col):
        """
        :param dataset: dataset, DataFrame
        :param file_col:  file name of the column
        :param label_col:  label name of the column
        """
        self.file = dataset[file_col].values
        self.label = dataset[label_col].values

        self.normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):

        label = torch.tensor(self.label[index])

        data = Image.open(self.file[index])
        data = self.normalize(data)


        return data, label

class VRDataset(Dataset):
    def __init__(self, data, label):
        """
        :param dataset: dataset, DataFrame
        :param file_col:  file name of the column
        :param label_col:  label name of the column
        """
        self.data = data

        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = torch.tensor(self.data[index])
        label = torch.tensor(self.label[index])

        return data, label

class PairsImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                index2 = random.randint(0, len(self.dataset) - 1)
                img2, label2 = self.dataset[index2]
                if label2 == label1:
                    break
        else:
            while True:
                index2 = random.randint(0, len(self.dataset) - 1)
                img2, label2 = self.dataset[index2]
                if label2 != label1:
                    break
        return img1, img2, label1, label2

    def __len__(self):
        return len(self.dataset)

class PairTabularDataset(Dataset):

    def __init__(self, dataset):
        """
        :param dataset: MyTabularDataset
        """
        self.dataset = dataset
        self.class_indices = self._create_class_indices()

    def __len__(self):
        return len(self.dataset)

    def _create_class_indices(self):
        """
        Precompute indices for each class in the dataset.
        This allows efficient sampling of same-class or different-class pairs.
        """
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset):
            class_indices[label.item()].append(idx)
        return class_indices

    def __getitem__(self, index):
        data_1, label_1 = self.dataset[index]

        # Determine if we should return a pair from the same class or a different class
        should_get_same_class = random.randint(0, 1)

        # Handle the case where there's only one class in the dataset
        if len(self.class_indices) == 1:
            should_get_same_class = 1

        if should_get_same_class:
            # Sample from the same class
            indices_same_class = self.class_indices[label_1.item()]
            if len(indices_same_class) > 1:
                index_2 = random.choice([i for i in indices_same_class if i != index])
            else:
                index_2 = index  # Fallback to itself if there's only one example of this class
        else:
            # Sample from a different class
            different_class_labels = list(self.class_indices.keys())
            different_class_labels.remove(label_1.item())  # Remove current class

            # Add a fallback in case there are no other classes available
            if different_class_labels:
                label_2 = random.choice(different_class_labels)
                index_2 = random.choice(self.class_indices[label_2])
            else:
                # If no other class is available, fallback to same class
                indices_same_class = self.class_indices[label_1.item()]
                if len(indices_same_class) > 1:
                    index_2 = random.choice([i for i in indices_same_class if i != index])
                else:
                    index_2 = index  # Fallback to itself if there's only one example of this class

        data_2, label_2 = self.dataset[index_2]

        return data_1, data_2, label_1, label_2


#def get_dataset(conf, data, load_pair_data):
#    """
#    :param conf: Configuration
#    :param data: Data (DataFrame)
#    :return:
#    """
#    if conf['data_type'] == 'tabular':
#        dataset = MyTabularDataset(data, conf['label_column'])
#        if load_pair_data:
#            dataset = PairTabularDataset(dataset=dataset)
#
#    elif conf['data_type'] == 'image':
#        
#        
#        dataset = MyImageDataset(data, conf['data_column'], conf['label_column'])
#        if load_pair_data:
#            dataset = PairsImageDataset(dataset)
#    else:
#        return None
#    return dataset


from torch.utils.data import Dataset

class MySyntheticDataset(Dataset):
    def __init__(self, dataset, label_col):
        """
        Initialize the synthetic dataset with raw pixel data.
        
        :param dataset: The DataFrame containing the data and labels.
        :param label_col: The column containing the labels.
        """
        self.data = dataset.drop(label_col, axis=1).values  # Pixel values (flattened)
        self.labels = dataset[label_col].values             # Labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Reshape the data assuming 32x32x3 image format for CIFAR-like data
        sample = self.data[idx].astype(np.float32).reshape(32, 32, 3)
        label = self.labels[idx]

        # Convert to PyTorch tensor
        sample = torch.tensor(sample).permute(2, 0, 1)  # Change to (C, H, W) format for PyTorch
        label = torch.tensor(label).long()

        return sample, label

#v2
def get_dataset(conf, data, load_pair_data):
    """
    :param conf: Configuration
    :param data: Data (DataFrame)
    :return: Dataset instance
    """
    # For tabular data
    if conf['data_type'] == 'tabular':
        dataset = MyTabularDataset(data, conf['label_column'])
        if load_pair_data:
            dataset = PairTabularDataset(dataset=dataset)

    # For image data
    elif conf['data_type'] == 'image':
        # Check if the dataset is synthetic
        if conf["dataset_used"] == "synthetic":
            dataset = MySyntheticDataset(data, conf['label_column'])
        else:
            dataset = MyImageDataset(data, conf['data_column'], conf['label_column'])
        
        if load_pair_data:
            dataset = PairsImageDataset(dataset)

    else:
        return None

    return dataset






