import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CFDataset(Dataset):
    def __init__(
            self,
            data_path: Path,
            train: bool = True,
            load_img2mem: bool = True,
            augment: bool = True,
        ):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.load_img2mem = load_img2mem
        img_load_fn = (lambda x: self.load_img(x)) if self.load_img2mem else (lambda x: x)

        if train and augment:
            self.tf = transforms.Compose([
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.1),
                transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=None),
                # transforms.Grayscale(num_output_channels=3),
                # transforms.RandomHorizontalFlip(),
                # transforms.Resize((40, 40), antialias=None),
                # transforms.RandomCrop((32, 32)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])

        self.x = []
        self.y = []
        self.cls_cnt = {}
        for cls_id in os.listdir(data_path):
            cls_path = data_path / cls_id
            if cls_id not in self.cls_cnt:
                self.cls_cnt[cls_id] = 0

            for img_name in os.listdir(cls_path):
                img_path = cls_path / img_name
                self.x.append(img_load_fn(img_path))
                self.y.append(int(cls_id))
                self.cls_cnt[cls_id] += 1

        self.cls_cnt = sorted(self.cls_cnt.items(), key=lambda x: x[0])
        print(f"Load {len(self)} images from {data_path} for {'train' if train else 'test'}")
        print(f"Class count: {self.cls_cnt}")


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_load_fn = (lambda x: x) if self.load_img2mem else (lambda x: self.load_img(x))
        return self.tf(img_load_fn(self.x[idx])), self.y[idx]

    def load_img(self, img_path):
        """
        Output:
            - img: torch.Tensor (3, h, w) in range [-1, 1]
        """
        img = Image.open(img_path).convert('RGB')
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img

if __name__ == "__main__":
    balacned_data_path = Path("data/CIFAR10_balance")
    imbalanced_data_path = Path("data/CIFAR10_imbalanced")

    b_dataset = CFDataset(balacned_data_path, load_img2mem=False)
    ib_dataset = CFDataset(imbalanced_data_path, load_img2mem=False)

    counter = {i: 0 for i in range(10)}
    for y in b_dataset.y:
        counter[y] += 1
    print(f"Balanced dataset: {counter}")

    counter.clear()
    counter = {i: 0 for i in range(10)}
    for y in ib_dataset.y:
        counter[y] += 1
    print(f"Imbalanced dataset: {counter}")

    """
    Balanced dataset: {0: 487, 1: 487, 2: 487, 3: 487, 4: 487, 5: 487, 6: 487, 7: 487, 8: 487, 9: 487}
    Imbalanced dataset: {0: 4435, 1: 4189, 2: 3065, 3: 4191, 4: 4513, 5: 3717, 6: 3513, 7: 3249, 8: 2685, 9: 2023}
    not so imbalanced midai
    """

    x, y = b_dataset[0]
    print(x.shape, x.min(), x.max(), x.mean(), x.std(), y)
    """
    torch.Size([3, 32, 32]) tensor(-1.) tensor(1.) tensor(0.0202) tensor(0.5498) 2
    """