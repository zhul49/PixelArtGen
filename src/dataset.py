import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PokemonDataset(Dataset):
    def __init__(self, artwork_dir, sprite_dir, image_size=64, augment=True):
        self.augment = augment
        self.image_size = image_size
        self.pairs = []

        # find pokemon that have both artwork and a sprite
        artwork_ids = set(f.replace('.png', '') for f in os.listdir(artwork_dir) if f.endswith('.png'))
        sprite_ids = set(f.replace('.png', '') for f in os.listdir(sprite_dir) if f.endswith('.png'))
        paired_ids = artwork_ids & sprite_ids

        for pid in sorted(paired_ids, key=lambda x: int(x) if x.isdigit() else float('inf')):
            self.pairs.append((
                os.path.join(artwork_dir, f'{pid}.png'),
                os.path.join(sprite_dir, f'{pid}.png')
            ))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        print(f"Found {len(self.pairs)} paired images")

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        # fill transparent background with black
        img = Image.open(path).convert('RGBA')
        bg = Image.new('RGBA', img.size, (0, 0, 0, 255))
        bg.paste(img, mask=img.split()[3])
        return bg.convert('RGB')

    def __getitem__(self, idx):
        artwork_path, sprite_path = self.pairs[idx]
        try:
            artwork = self._load_image(artwork_path)
            sprite = self._load_image(sprite_path)

            if self.augment:
                # flip both images together so the pair stays consistent
                if random.random() > 0.5:
                    artwork = F.hflip(artwork)
                    sprite = F.hflip(sprite)

                # vary artwork colors slightly to help generalize
                if random.random() > 0.5:
                    artwork = F.adjust_brightness(artwork, random.uniform(0.8, 1.2))
                    artwork = F.adjust_contrast(artwork, random.uniform(0.8, 1.2))
                    artwork = F.adjust_saturation(artwork, random.uniform(0.8, 1.2))

            return self.transform(artwork), self.transform(sprite)

        except Exception:
            return self.__getitem__((idx + 1) % len(self.pairs))


if __name__ == "__main__":
    dataset = PokemonDataset(
        artwork_dir='data/pokemon/sprites/pokemon/other/official-artwork',
        sprite_dir='data/pokemon/sprites/pokemon'
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    artwork_batch, sprite_batch = next(iter(dataloader))
    print(f"Artwork batch shape: {artwork_batch.shape}")
    print(f"Sprite batch shape: {sprite_batch.shape}")
    print(f"Pixel range: {sprite_batch.min():.2f} to {sprite_batch.max():.2f}")
