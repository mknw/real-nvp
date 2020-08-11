import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def load_mnist(args, test=False):
    if not test:
        transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.ColorJitter(brightness=0.3)
        ])
        trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
        loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
        loader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return testloader

def load_celeba(batch, image_size, test=False):
    if not test:
        split = 'train'
        transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])
    else:
        split = 'test'
        transform = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Resize(size=image_size),
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])
    target_type = ['attr', 'bbox', 'landmarks']
    dataset = datasets.CelebA(root='data', split=split, target_type=target_type[0], download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=8)
    return loader

def load_cifar10(args, test=False):
    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    if not test:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    dataset = torchvision.datasets.CIFAR10(root='data', train=(not test), download=True, transform=transform_train)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader


def sample_celeba(batch, image_size, test=False):
    if not test:
        split = 'train'
        transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize(size=image_size),
            transforms.RandomHorizontalFlip(),
            # TODO: add conditional based on resize_hw_
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])
    else:
        split = 'test'
        transform = transforms.Compose([
        transforms.CenterCrop(100),
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])
    target_type = ['attr', 'bbox', 'landmarks']
    dataset = datasets.CelebA(root='data', split=split, target_type=target_type[0], download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=8)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=8)
            loader = iter(loader)
            yield next(loader)
