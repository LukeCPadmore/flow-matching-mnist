import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

default_transform = transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(2, padding_mode="constant"),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def create_mnist_train_val_loaders(
    batch_size: int = 64,
    root: str = "/home/luke-padmore/Source/flow-matching-mnist/data",
    transform: transforms.Compose = default_transform,
    num_workers=4,
    shuffle=True,
) -> tuple[DataLoader, DataLoader]:
    trainset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_set = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return train_loader, val_loader
