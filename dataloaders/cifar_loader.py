import torch.utils.data as data
import os
import numpy as np
import torch
import hashlib
from PIL import Image
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    def __init__(self, root, per_class_labeled_image, per_class_unlabeled_image, train=True, finetune=False, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training or testing
        self.finetune = finetune  # self supervision or fine-tuning

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.array(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.train:
            assert per_class_labeled_image + per_class_unlabeled_image < 5000, "per_class_labeled_image+per_class_unlabeled_image>5000"

            select_index_set = np.zeros(len(self.class_to_idx) * per_class_labeled_image, dtype=np.int) - 1
            label_counter = [0] * len(self.class_to_idx)
            j = 0
            for i, label in enumerate(self.targets):
                if label_counter[label] != per_class_labeled_image:
                    label_counter[label] += 1
                    select_index_set[j] = i
                    j += 1
                if label_counter == [per_class_labeled_image] * len(self.class_to_idx):
                    break
            unselected_index_set = np.ones(self.targets.shape).astype(np.bool)
            unselected_index_set[select_index_set] = 0
            unselected_index_set, = np.where(unselected_index_set)

            if self.finetune:
                self.data = self.data[select_index_set]
                self.targets = self.targets[select_index_set]
            else:
                rest_data = self.data[unselected_index_set]
                rest_targets = self.targets[unselected_index_set]
                select_index_set = np.zeros(len(self.class_to_idx) * per_class_unlabeled_image, dtype=np.int) - 1
                label_counter = [0] * len(self.class_to_idx)
                j = 0
                for i, label in enumerate(rest_targets):
                    if label_counter[label] != per_class_unlabeled_image:
                        label_counter[label] += 1
                        select_index_set[j] = i
                        j += 1
                    if label_counter == [per_class_unlabeled_image] * len(self.class_to_idx):
                        break
                self.data = rest_data[select_index_set]
                self.targets = rest_targets[select_index_set].tolist()
                # we simply shift every label to the right by 1, it is equivalent to a deterministic shuffle
                self.targets.insert(0, self.targets[-1])
                self.targets.pop(-1)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


class CIFAR10_Contrast(data.Dataset):
    """
    This dataloader is for contrastive learning, it applies transformation to the same image twice and returns them
    """
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    def __init__(self, root, train, transform, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        img1 = self.transform(img)
        img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img1, img2], target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


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
        # if isinstance(img, np.ndarray):
        #     img = Image.fromarray(np.uint8(img))
        # elif isinstance(img, torch.ByteTensor):
        #     img = Image.fromarray(img.numpy())
        # else:
        #     raise Exception('img must be torch.ByteTensor or np.ndarray')

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


class CIFAR_Return_Original_Image(data.Dataset):
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
        # if isinstance(img, np.ndarray):
        #     img = Image.fromarray(np.uint8(img))
        # elif isinstance(img, torch.ByteTensor):
        #     img = Image.fromarray(img.numpy())
        # else:
        #     raise Exception('img must be torch.ByteTensor or np.ndarray')

        augmented_img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index:
            return augmented_img, img, target, index
        else:
            return augmented_img, img, target

    def __len__(self):
        return len(self.data)


def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return img.rot90(1, [1, 2])
    elif rot == 180:  # 90 degrees rotation
        return img.rot90(2, [1, 2])
    elif rot == 270:  # 270 degrees rotation / or -90
        return img.rot90(1, [2, 1])
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class CIFAR_RotNet(data.Dataset):
    '''
    Dataloader for RotNet
    the image first goes through data augmentation, and then rotate 4 times
    the output is 4 rotated views of the augmented image,
    the corresponding labels are 0 1 2 3
    '''
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        return img, rotate_img(img, 90), rotate_img(img, 180), rotate_img(img, 270), rotation_labels

    def __len__(self):
        return len(self.data)


class CIFAR_Partial_CELoss(data.Dataset):
    '''
    return one-hot targets, g_mask and a indicator
    indicator is a list of int
    indicator is 1 meaning targets is the correct class
    indicator is 1 meaning targets are the incorrect class
    targets is a list of lists (because for incorrect class, there are more than 1 labels)

    e.g.:
    indicators: [1,0,1,0]
    targets: [[0], [0,1], [2], [1,2]]
    then y_onehot after collecting all data should be:
    [[1,0,0],
     [0,0,0],
     [0,0,1],
     [0,0,0]
    ]
    g_mask should be:
    [[1,1,1],
     [1,1,0],
     [1,1,1],
     [0,1,1]
    ]
    then backward would be:
    for x, y_onehot, g_mask, _ in dataloader:
        out = model(x)
        p = F.softmax(out, dim=-1)
        g = p - y_onehot
        g_mask /= g_mask.sum(dim=0)
        g *= g_mask

        optimizer.zero_grad()
        torch.autograd.backward([out], g)
        optimizer.step()

    If indicators are all 1, then we can also convert this back to standard CE loss:
    for x, y_onehot, _, _ in dataloader:
        targets = y_onehot.nonzero()[:, 1]
        assert len(targets) == y_onehot.size(0), 'correct class has more than 1 label!!!'
        out = model(x)
        loss = criterion(out, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    '''
    def __init__(self, data, targets, discourage_targets, num_classes, min_all=0,
                 transform=None, target_transform=None, return_index=False):
        assert len(targets) == len(discourage_targets), '#targets not equal to #discourage_targets!!!'
        self.data = data
        self.targets = targets
        self.discourage_targets = discourage_targets
        # self.indicators = indicators
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes
        self.min_all = min_all
        self.return_index = return_index
        # for i in indicators:
        #     assert len(targets[i]) == 1, str(i) + ' has more than 1 label but indicator is 1'

    def __getitem__(self, index):
        img, target, discourage_targets = self.data[index], self.targets[index], self.discourage_targets[index]
        # img = Image.fromarray(img.numpy())
        if self.transform is not None:
            img = self.transform(img)

        # if indicator == 1:
        #     y_onehot = torch.FloatTensor(self.num_classes).zero_()
        #     g_mask = torch.FloatTensor(self.num_classes).zero_()
        #     assert len(target) == 1
        #     y_onehot[target] = 1
        #     g_mask.fill_(1)
        y_onehot = torch.FloatTensor(self.num_classes).zero_()
        g_mask = torch.FloatTensor(self.num_classes).zero_()
        if self.min_all:
            g_mask.fill_(1)
        else:
            g_mask[discourage_targets] = 1
        if self.return_index:
            return img, y_onehot, g_mask, target, index
        else:
            return img, y_onehot, g_mask, target

    def __len__(self):
        return len(self.data)


class CIFAR_RCELoss(data.Dataset):
    def __init__(self, data, targets, indicators, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.indicators = indicators
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target, indicator = self.data[index], self.targets[index], self.indicators[index]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(np.uint8(img))
        elif isinstance(img, torch.ByteTensor):
            img = Image.fromarray(img.numpy())
        else:
            raise Exception('img must be torch.ByteTensor or np.ndarray')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, indicator

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import torchvision.datasets as datasets
    class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
                    'ship': 8, 'truck': 9}
    idx2class = {value:key for key, value in class_to_idx.items()}

    tmp = datasets.CIFAR10(root='/BS/databases00/cifar-10', train=True)
    num_classes = 10
    total_data = torch.tensor(tmp.data, dtype=torch.uint8)
    total_targets = torch.tensor(tmp.targets, dtype=torch.int64)

    select_index_set = np.zeros(num_classes * 3, dtype=np.int) - 1
    label_counter = [0] * num_classes
    j = 0
    for i, label in enumerate(total_targets):
        if label_counter[label] != 3:
            label_counter[label] += 1
            select_index_set[j] = i
            j += 1
        if label_counter == 3 * num_classes:
            break
    unselected_index_set = np.ones(total_targets.shape).astype(np.bool)
    unselected_index_set[select_index_set] = 0
    unselected_index_set, = np.where(unselected_index_set)

    selected_data = total_data[select_index_set]
    selected_targets = total_targets[select_index_set]
    unselected_data = total_data[unselected_index_set]
    unselected_targets = total_targets[unselected_index_set]

    # data of what it is not
    index_set = np.zeros(num_classes * 3, dtype=np.int) - 1
    label_counter = [0] * num_classes
    j = 0
    for i, label in enumerate(unselected_targets):
        if label_counter[label] != 3:
            label_counter[label] += 1
            index_set[j] = i
            j += 1
        if label_counter == [3] * num_classes:
            break
    discourage_data = unselected_data[index_set]
    discourage_targets = unselected_targets[index_set]
    # shuffle the discourage_targets so that all labels are wrong
    # while 1:
    #     shuffled_index = torch.randperm(len(discourage_targets))
    #     if not (0 in shuffled_index - torch.arange(len(discourage_targets))):  # no correct label in corrupt_targets
    #         break
    # discourage_targets = discourage_targets[shuffled_index]
    d = []
    for label in discourage_targets:
        incorrect_label = [int(label) - 2, int(label) - 1, int(label) + 1, int(label) + 2]
        for i, data in enumerate(incorrect_label):
            if data < 0:
                incorrect_label[i] = data + num_classes
            if data >= num_classes:
                incorrect_label[i] = data - num_classes

        d.append(incorrect_label)

    num_correct = len(selected_data)
    num_incorrect = len(discourage_data)
    selected_data = torch.cat([selected_data, discourage_data], 0)
    # selected_targets = torch.cat([selected_targets, discourage_targets], 0)
    # labeled_targets = [[int(i)] for i in selected_targets]
    labeled_targets = [[int(i)] for i in selected_targets] + d

    indicators = [1] * num_correct + [0] * num_incorrect
    print(num_correct, num_incorrect)

    loader = CIFAR_OneHot(selected_data, labeled_targets, indicators, 10)

    print(len(loader))

    import matplotlib.pyplot as plt
    import numpy as np

    ii = [0,10,20,30,40,50]
    for i in ii:
        img, y_onehot, g_mask, indicator = loader[i]
        plt.imshow(np.asarray(img))
        plt.show()
        print(y_onehot)
        print(g_mask)
        print(indicator)
        if indicator == 1:
            label = int(y_onehot.nonzero()[0][0])
        else:
            label = int(g_mask.nonzero()[0][0])
        print(idx2class[label])

