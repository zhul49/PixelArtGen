import os
import sys
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import PokemonDataset
from PIL import Image


def make_verification_grid(dataset, output_path='pair_verification.png'):
    n_pairs = len(dataset.pairs)
    cols = 10
    rows = math.ceil(n_pairs / (cols // 2))
    cell_size = 64
    padding = 2

    grid_w = cols * (cell_size + padding) + padding
    grid_h = rows * (cell_size + padding) + padding
    grid = Image.new('RGB', (grid_w, grid_h), (40, 40, 40))

    for i in range(n_pairs):
        artwork_path, sprite_path = dataset.pairs[i]
        try:
            # load artwork with black background
            artwork_raw = Image.open(artwork_path).convert('RGBA')
            bg = Image.new('RGBA', artwork_raw.size, (0, 0, 0, 255))
            bg.paste(artwork_raw, mask=artwork_raw.split()[3])
            artwork = bg.convert('RGB').resize((cell_size, cell_size))

            # load sprite with black background
            sprite_raw = Image.open(sprite_path).convert('RGBA')
            bg = Image.new('RGBA', sprite_raw.size, (0, 0, 0, 255))
            bg.paste(sprite_raw, mask=sprite_raw.split()[3])
            sprite = bg.convert('RGB').resize((cell_size, cell_size))
        except Exception:
            continue

        row = i // (cols // 2)
        col = (i % (cols // 2)) * 2

        x_art = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + padding)
        x_spr = padding + (col + 1) * (cell_size + padding)

        grid.paste(artwork, (x_art, y))
        grid.paste(sprite, (x_spr, y))

    grid.save(output_path)
    print(f"Saved {output_path} — {n_pairs} pairs, artwork left, sprite right")


if __name__ == "__main__":
    dataset = PokemonDataset(
        artwork_dir='data/pokemon/sprites/pokemon/other/official-artwork',
        sprite_dir='data/pokemon/sprites/pokemon'
    )
    make_verification_grid(dataset)
