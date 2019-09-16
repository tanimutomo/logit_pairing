import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

# -------------------------------------------------------------
#    Helpers
# -------------------------------------------------------------

def load_mnist(batch_size, data_dir, augmentation=False, stddev=0.0, adv_subset=1000, workers=4):

    trainloader, _, classes = get_mnist(batch_size=batch_size,
                                        train=True,
                                        path=data_dir,
                                        augmentation=augmentation,
                                        std=stddev,
                                        shuffle=True,
                                        workers=workers
                                        )

    testloader, _, _ = get_mnist(batch_size=batch_size,
                                 train=False,
                                 path=data_dir,
                                 shuffle=False,
                                 workers=workers
                                 )

    adv_testloader, _, _ = get_mnist(batch_size=batch_size,
                                     train=False,
                                     path=data_dir,
                                     shuffle=False,
                                     adversarial=True,
                                     subset=adv_subset,
                                     workers=workers
                                     )

    input_shape = (None, 28, 28, 1)

    return trainloader, testloader, adv_testloader, input_shape, len(classes)


def load_cifar10(batch_size, data_dir, augmentation=False, stddev=0.0, adv_subset=1000, workers=4):

    trainloader, _, classes = get_cifar10(batch_size=batch_size,
                                          train=True,
                                          path=data_dir,
                                          augmentation=augmentation,
                                          std=stddev,
                                          shuffle=True,
                                          workers=workers
                                          )

    testloader, _, _ = get_cifar10(batch_size=batch_size,
                                   train=False,
                                   path=data_dir,
                                   shuffle=False,
                                   workers=workers
                                   )

    adv_testloader, _, _ = get_cifar10(batch_size=batch_size,
                                       train=False,
                                       path=data_dir,
                                       shuffle=False,
                                       adversarial=True,
                                       subset=adv_subset,
                                       workers=workers
                                       )

    input_shape = (None, 32, 32, 3)

    return trainloader, testloader, adv_testloader, input_shape, len(classes)



def get_mnist(batch_size, train, path, augmentation=False, std=0.0, shuffle=True, adversarial=False, subset=1000,
              workers=0):
    classes = np.arange(0, 10)

    if augmentation:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.Tensor(x.size()).normal_(mean=0.0, std=std)),  # add gaussian noise
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

    dataset = torchvision.datasets.MNIST(root=path,
                                         train=train,
                                         download=True,
                                         transform=transform)

    if adversarial:
        np.random.seed(123)  # load always the same random subset
        indices = np.random.choice(np.arange(dataset.__len__()), subset)

        subset_sampler = SubsetSampler(indices)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 sampler=subset_sampler,
                                                 num_workers=workers
                                                 )

    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=workers
                                                 )
    return dataloader, dataset, classes


def get_cifar10(batch_size, train, path, augmentation=False, std=0.0, shuffle=True, adversarial=False, subset=1000,
                workers=0):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if augmentation:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.Tensor(x.size()).normal_(0.0, std)),  # add gaussian noise
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

    dataset = torchvision.datasets.CIFAR10(root=path,
                                           train=train,
                                           download=True,
                                           transform=transform)

    if adversarial:
        np.random.seed(123)  # load always the same random subset
        indices = np.random.choice(np.arange(dataset.__len__()), subset)

        subset_sampler = SubsetSampler(indices)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 sampler=subset_sampler,
                                                 num_workers=workers
                                                 )
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=workers
                                                 )
    return dataloader, dataset, classes


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
        self.shuffle = False

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
