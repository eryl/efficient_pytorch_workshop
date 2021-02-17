import argparse
from pathlib import Path
from collections import defaultdict
import math

import numpy as np

from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from torchvision.models import resnet152, resnet18, alexnet
from torchvision.datasets.folder import VisionDataset, ImageFolder, default_loader, make_dataset
import matplotlib.pyplot as plt
import deepspeed

from tqdm import tqdm

try:
    @profile
    def foo():
        pass
    del foo
except NameError:
    def profile(f):
        return f



class ImageDatasetFromLists(VisionDataset):
    def __init__(self, images, labels, *args, **kwargs):
        self.images = images
        self.labels = labels
        self.num_classes = len(np.unique(self.labels))
        super(ImageDatasetFromLists, self).__init__(*args, **kwargs)

    def __getitem__(self, item):
        path, label = self.images[item]
        image = default_loader(path)
        return self.transform(image), label

    def __len__(self):
        return len(self.labels)



class LogisticRegression(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LogisticRegression, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.input_dim = np.prod(input_shape)
        self.W = nn.Linear(self.input_dim, num_classes, bias=True)

    def forward(self, x):
        x = x.view((-1, self.input_dim))
        return self.W(x)



def make_datasets(dataset_path: Path, dev_ratio=0.1, test_ratio=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # find images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((224, 224))
    ])
    dataset = ImageFolder(dataset_path)
    imgs = dataset.imgs
    labels = dataset.targets
    label_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_indices[label].append(i)

    train_indices = []
    dev_indices = []
    test_indices = []

    for label, indices in label_indices.items():
        n = len(indices)
        n_dev = int(math.ceil(n * dev_ratio))
        n_test = int(math.ceil(n * test_ratio))
        n_train = n - (n_dev + n_test)
        rng.shuffle(indices)
        train_indices.extend(indices[:n_train])
        dev_indices.extend(indices[n_train:n_train+n_dev])
        test_indices.extend(indices[n_train+n_dev:])

    train_dataset = ImageDatasetFromLists([imgs[i] for i in train_indices], [labels[i] for i in train_indices], root=dataset_path,
                                          transform=transform)
    dev_dataset = ImageDatasetFromLists([imgs[i] for i in dev_indices], [labels[i] for i in dev_indices], root=dataset_path,
                                        transform=transform)
    test_dataset = ImageDatasetFromLists([imgs[i] for i in test_indices], [labels[i] for i in test_indices], root=dataset_path,
                                         transform=transform)

    return train_dataset, dev_dataset, test_dataset


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Simple Resnet training examaple')
    parser.add_argument('images', type=Path)
    parser.add_argument('--pipeline-batch', default=False, action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pin-memory', default=False, action='store_true')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--prefetch-factor', type=int)
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    rng = np.random.default_rng(1729)

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available and device set to {args.device}")
        else:
            device = torch.device(args.device)

    print(f"Device is set to {device}")

    train_set, dev_set, test_set = make_datasets(args.images, rng=rng)

    batch_size = 8
    max_epochs = 1

    dataloader_kwargs = dict()
    if args.pin_memory:
        dataloader_kwargs['pin_memory'] = True
    if args.num_workers:
        dataloader_kwargs['num_workers'] = args.num_workers
    if args.prefetch_factor:
        dataloader_kwargs['prefetch_factor'] = args.num_workers

    model = resnet152(pretrained=False, num_classes=train_set.num_classes)
    #model = resnet18(pretrained=False, num_classes=train_set.num_classes)
    #model = LogisticRegression(train_set[0][0].shape, train_set.num_classes)
    #model = alexnet(pretrained=False, num_classes=train_set.num_classes)

    model_engine, optimizer, _, __ = deepspeed.initialize(args=args,
                                                          model=model,
                                                          model_parameters=model.parameters())
    training_loader = model_engine.deepspeed_io(train_set, **dataloader_kwargs)
    dev_loader = model_engine.deepspeed_io(dev_set, **dataloader_kwargs)
    test_loader = model_engine.deepspeed_io(test_set, **dataloader_kwargs)

    loss_fn = nn.CrossEntropyLoss()
    #optimizer = AdamW(model.parameters(), lr=1e-4)

    for epoch in range(max_epochs):
        training_losses = []
        model_engine.train()
        for x, y in tqdm(training_loader, desc='training progress', total=len(training_loader)):
            #optimizer.zero_grad()
            prediction = model_engine(x.to(model_engine.local_rank))
            loss = loss_fn(prediction, y.to(model_engine.local_rank))
            #loss.backward()
            #optimizer.step()
            model_engine.backward(loss)
            model_engine.step()
            training_losses.append(loss.item())
        print(f'Training loss: {np.mean(training_losses)}')

        val_losses = []
        model_engine.eval()
        with torch.no_grad():
            for x,y in dev_loader:
                prediction = model(x.to(model_engine.local_rank))
                loss = loss_fn(prediction, y.to(model_engine.local_rank))
                val_losses.append(loss.item())
        print(f'Validation loss: {np.mean(val_losses)}')

    test_match = []
    model_engine.eval()
    with torch.no_grad():
        for x,y in test_loader:
            prediction = model(x.to(model_engine.local_rank))
            correct = torch.argmax(prediction, dim=-1) == y.to(model_engine.local_rank)
            test_match.extend(correct.cpu().tolist())

    print(f'Test accuracy: {np.mean(test_match)}')


if __name__ == '__main__':
    main()

