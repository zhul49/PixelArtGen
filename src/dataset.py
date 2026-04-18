import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SpriteDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(root, file))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)


if __name__ == "__main__":
    dataset = SpriteDataset('data/raw/data/data')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    batch = next(iter(dataloader))
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch shape: {batch.shape}")
    print(f"Pixel value range: {batch.min():.2f} to {batch.max():.2f}")
