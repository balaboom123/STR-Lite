from typing import List, Sequence, Tuple

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def normalize_size_options(size_options) -> List[Tuple[int, int]]:
    if size_options is None:
        raise ValueError("size_options cannot be None")

    normalized: List[Tuple[int, int]] = []
    for item in size_options:
        if isinstance(item, int):
            normalized.append((item, item))
        elif isinstance(item, Sequence) and len(item) == 2:
            normalized.append((int(item[0]), int(item[1])))
        else:
            raise ValueError(f"Invalid size option: {item}")

    if not normalized:
        raise ValueError("size_options must contain at least one target size")
    return normalized


def build_str_transform(
    img_height: int,
    img_width: int,
    augment: bool = True,
    randaugment_layers: int = 2,
    randaugment_magnitude: int = 5,
    randaugment_prob: float = 1.0,
):
    ops = []

    if augment:
        if randaugment_layers > 0 and randaugment_prob > 0:
            rand_aug = transforms.RandAugment(
                num_ops=randaugment_layers,
                magnitude=randaugment_magnitude,
                interpolation=InterpolationMode.BILINEAR,
            )
            if randaugment_prob >= 1.0:
                ops.append(rand_aug)
            else:
                ops.append(transforms.RandomApply([rand_aug], p=randaugment_prob))

        ops.append(
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03)],
                p=0.5,
            )
        )
        ops.append(
            transforms.RandomAffine(
                degrees=2,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=1,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        )

    ops.extend(
        [
            transforms.Resize((img_height, img_width), interpolation=InterpolationMode.BICUBIC),
            # Keep CPU tensor as uint8 for faster host->device transfer.
            transforms.PILToTensor(),
        ]
    )

    return transforms.Compose(ops)
