from torch.utils.data import Subset
from torch import save
from torchvision import datasets, transforms
import os


def get_data(transform, cifar_flag):
    if cifar_flag:
        train_data = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='data/', train=False, download=True, transform=transform)
    else:
        train_data = datasets.FashionMNIST(root='data/', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='data/', train=False, download=True, transform=transform)
    return train_data, test_data

def sort_dataset(data):
    # Get unique classes
    unique_classes = list(data.class_to_idx.values())
    # Dict that gets the data indices for each unique label
    class_to_indices = {}
    for cl in unique_classes:
        class_to_indices[cl] = []
    
    for i in range(len(data)):
        class_to_indices[data.targets[i]].append(i)

    return class_to_indices

def create_subset(data, class_to_indices, classes, path):
    l = []
    for cl in classes:
        l.extend(class_to_indices[cl])
    sub = Subset(dataset=data, indices=l)
    print(len(sub.indices))
    save(obj=sub, f=path)

def save_subsets(subsets: dict, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for cl, sub in subsets.items():
        print('Storing subset for class', cl)
        save(obj=sub, f=f'{folder}/subset_{cl}.pth')


if __name__ == '__main__':
    if not os.path.exists('subset_data'):
        os.makedirs('subset_data')
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('models/client'):
        os.makedirs('models/client')
    if not os.path.exists('models/server'):
        os.makedirs('models/server')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # AlexNet tranformations
    #transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data, test_data = get_data(transform=transform, cifar_flag=True)
    class_to_indices = sort_dataset(data=train_data)
    create_subset(data=train_data, class_to_indices=class_to_indices, classes=[0], path='subset_data/sub_0.pth')
    create_subset(data=train_data, class_to_indices=class_to_indices, classes=[1], path='subset_data/sub_1.pth')
    create_subset(data=train_data, class_to_indices=class_to_indices, classes=[2], path='subset_data/sub_2.pth')
    create_subset(data=train_data, class_to_indices=class_to_indices, classes=[3], path='subset_data/sub_3.pth')