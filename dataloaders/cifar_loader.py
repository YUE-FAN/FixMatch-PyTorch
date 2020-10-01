import torch.utils.data as data


class CIFAR_Semi(data.Dataset):
    '''
    Just a wrapper datasets, for cifar10 and cifar100
    '''
    def __init__(self, data, targets, transform=None, target_transform=None, return_index=False):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = return_index

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)
