import os
import csv
import random
import numpy as np
import paddle
from paddle.vision.transforms import Resize, Compose, Normalize, RandomCrop, RandomRotation, RandomHorizontalFlip, ToTensor
from paddle.vision import transforms

from PIL import Image

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(paddle.nn.Layer):
    def __init__(self, f, p=1):
        super().__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(RandomHorizontalFlip(p=0.5))

    transforms_list.append(ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
        pass
    elif opt.dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        transforms_list.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class PostTensorTransform(paddle.nn.Layer):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop)
        self.random_rotation = transforms.RandomRotation(opt.random_rotation)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(prob=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
    
class GTSRB(paddle.io.Dataset):
    def __init__(self, opt, train, transforms, data_root=None, min_width=0):
        super(GTSRB, self).__init__()
        if data_root is None:
            data_root = opt.data_root
        if train:
            self.data_folder = os.path.join(data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list(min_width=min_width)
            if min_width > 0:
                print(f'Loading GTSRB Train greater than {min_width} width. Loaded {len(self.images)} images.')
        else:
            self.data_folder = os.path.join(data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list(min_width)
            print(f'Loading GTSRB Test greater than {min_width} width. Loaded {len(self.images)} images.')

        self.transforms = transforms

    def _get_data_train_list(self, min_width=0):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                if int(row[1]) >= min_width:
                    images.append(prefix + row[0])
                    labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self, min_width=0):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            if int(row[1]) >= min_width: #only load images if more than certain width
                images.append(self.data_folder + "/" + row[0])
                labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


def get_dataloader(opt, train=True, pretensor_transform=False, min_width=0):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        #print('HERE')
        dataset = GTSRB(opt, train, transform,min_width=min_width)
    elif opt.dataset == "mnist":
        dataset = paddle.vision.datasets.MNIST(
            mode='train' if train else 'test', transform=transform)
    elif opt.dataset == "cifar10":
        dataset = paddle.vision.datasets.Cifar10(
            mode='train' if train else 'test', transform=transform)
    elif opt.dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        if train:
            split = 'train'
        else:
            split = 'test'
        dataset = paddle.vision.datasets.ImageFolder(
            os.path.join(opt.data_root, 'tiny-imagenet-200', split), transform=transform)
    else:
        raise Exception("Invalid dataset")
    dataloader = paddle.io.DataLoader(
        dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    return dataloader


def get_dataset(opt, train=True):
    if opt.dataset == "gtsrb":
        dataset = GTSRB(
            opt,
            train,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset == "mnist":
        dataset = paddle.vision.datasets.MNIST(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "cifar10":
        dataset = paddle.vision.datasets.Cifar10(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        if train:
            split = 'train'
        else:
            split = 'test'
        dataset = paddle.vision.datasets.ImageFolder(
            os.path.join(opt.data_root, 'tiny-imagenet-200', split), 
            transform=paddle.vision.transforms.Compose([Resize((opt.input_height, opt.input_width)), ToNumpy()]))
    else:
        raise Exception("Invalid dataset")
    return dataset


def main():
    pass


if __name__ == "__main__":
    main()
