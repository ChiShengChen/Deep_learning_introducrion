# Custom Dataset, DataLoader and Preprocessing:
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Example data
data = ['image1.jpg', 'image2.jpg']
labels = [0, 1]

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = CustomDataset(data, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for images, labels in dataloader:
    print(images.shape, labels)
