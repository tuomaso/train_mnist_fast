import torch
from torchvision import datasets, transforms

def save_data():
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=10000, shuffle=True)

    train_data = []
    train_labels = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        train_data.append(data)
        train_labels.append(target)

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    torch.save(train_data, 'data/train_data.pt')
    torch.save(train_labels, 'data/train_labels.pt')
    
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=10000, shuffle=True)

    test_data = []
    test_labels = []

    for batch_idx, (data, target) in enumerate(test_loader):
        test_data.append(data)
        test_labels.append(target)

    test_data = torch.cat(test_data, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    torch.save(test_data, 'data/test_data.pt')
    torch.save(test_labels, 'data/test_labels.pt')