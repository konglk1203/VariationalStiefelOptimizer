import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from autoaugment import CIFAR10Policy, SVHNPolicy
from criterions import LabelSmoothingCrossEntropyLoss

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_model(args):        
    from ViT_non_vectorized import ViT_non_vectorized
    if args.constraint=='OnlyWithin' and args.optim_method in ['ProjectedStiefelSGD', 'ProjectedStiefelAdam']:
        net = ViT_non_vectorized(
            channels=args.in_c, 
            num_classes=args.num_classes, 
            image_size=args.size, 
            patch_size=args.size//args.patch, 
            dropout=args.dropout, 
            mlp_dim=args.mlp_hidden,
            depth=args.num_layers,
            dim=args.hidden,
            heads=args.head,
            dim_head=args.hidden//args.head,
            pool='cls'
        )
    else:
        from ViT import ViT
        net = ViT(
            channels=args.in_c, 
            num_classes=args.num_classes, 
            image_size=args.size, 
            patch_size=args.size//args.patch, 
            dropout=args.dropout, 
            mlp_dim=args.mlp_hidden,
            depth=args.num_layers,
            dim=args.hidden,
            heads=args.head,
            dim_head=args.hidden//args.head,
            pool='cls',
            constraint=args.constraint
        )
    return net


def get_transform(args):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=args.size, padding=args.padding)
    ]
    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = "../data_set"
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(root, split="train",transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds