import os
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio
from skimage.transform import rotate, warp
from torchvision import transforms


class AngioDB(Dataset):
    def __init__(self, root, mode='train', transform=None, size=-1):
        self._transform = transform

        if 'train' in mode:
            self._data_fn = list(map(lambda i: os.path.join(root, '{}.png'.format(i+1)), range(75)))
            self._target_fn = list(map(lambda i: os.path.join(root, '{}_gt.png'.format(i+1)), range(75)))

        elif 'valid' in mode:
            self._data_fn = list(map(lambda i: os.path.join(root, '{}.png'.format(i + 1)), range(75, 100)))
            self._target_fn = list(map(lambda i: os.path.join(root, '{}_gt.png'.format(i + 1)), range(75, 100)))

        elif 'test' in mode:
            self._data_fn = list(map(lambda i: os.path.join(root, '{}.png',format(i+1)), range(100, 130)))
            self._target_fn = list(map(lambda i: os.path.join(root, '{}_gt.png'.format(i+1)), range(100, 130)))

        if size > 0:
            self._sample_idx = np.random.choice(len(self._data_fn), size, replace=size > len(self._data_fn))
        else:
            self._sample_idx = np.arange(len(self._data_fn))

        self._data = list(map(imageio.imread, self._data_fn))
        self._target = list(map(imageio.imread, self._target_fn))

    def __len__(self):
        return self._sample_idx.size

    def __getitem__(self, item):
        i = self._sample_idx[item]

        if self._transform is not None:
            data_target = np.concatenate((self._data[i][..., np.newaxis], self._target[i][..., np.newaxis]), axis=2)
            data_target = self._transform(data_target)
            if isinstance(data_target, np.ndarray):
                data = torch.from_numpy(data_target[..., 0]).float()
                target = torch.from_numpy(data_target[..., 1]).float()
            else:
                data = data_target[0, ...].float()
                target = data_target[1, ...].float()

        else:
            target = self._data[i]
            data = self._target[i]
            data = torch.from_numpy(data).float()
            target = torch.from_numpy(target).float()

        data = (data - data.min()) / (data.max() - data.min())
        return data[np.newaxis, ...], target


class RandFlip(object):
    def __init__(self, prob_h=0.5, prob_v=0.5):
        self._prob_h = prob_h
        self._prob_v = prob_v

    def __call__(self, img):
        if np.random.rand() < self._prob_h:
            flipped_img = img[::-1, ...].copy()
        else:
            flipped_img = img.copy()

        if np.random.rand() < self._prob_v:
            flipped_img = flipped_img[:, ::-1, ...].copy()

        return flipped_img


class RandGamma(object):
    def __init__(self, prob=0.5, min_gamma=1.0, max_gamma=1.0):
        self._prob = prob
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma

    def __call__(self, img):
        if np.random.rand() > self._prob:
            return img

        rand_gamma = np.random.rand() * (self._max_gamma - self._min_gamma) + self._min_gamma
        corrected_img = np.float32(img[..., :-1])
        min_intensity = corrected_img.min()
        rng_intensity = corrected_img.max() - min_intensity
        corrected_img = (corrected_img - min_intensity) / rng_intensity
        corrected_img = np.uint8(corrected_img ** rand_gamma * rng_intensity + min_intensity)

        return np.concatenate((corrected_img, img[..., -1, np.newaxis]), axis=2)


class RandomRotation(object):
    """Rotate the image and its respective target by a random angle.

    Args:
        center_offset: Allow to move the center of reference on the image a random offset.
    """

    def __init__(self, prob=0.5, center_offset=False):
        self._prob = prob
        self._center_offset = center_offset

    def __call__(self, img):
        if np.random.rand() > self._prob:
            return img

        h, w = img.shape[:2]

        center_h = int((np.random.rand()*2-1) * h//4 + h//2) if self._center_offset else h // 2
        center_w = int((np.random.rand()*2-1) * w//4 + w//2) if self._center_offset else w // 2
        angle = np.random.rand() * 360.0

        data = img[..., :-1]
        target = img[..., -1]
        min_intensity = data.min()
        rng_intensity = data.max() - min_intensity

        rot_image = np.uint8(rotate(data, angle, False, [center_w, center_h], order=3) * rng_intensity + min_intensity)
        rot_target = np.uint8(rotate(target, angle, False, [center_w, center_h], order=0)*255.0)

        return np.concatenate((rot_image, rot_target[..., np.newaxis]), axis=2)


class RandomShear(object):
    def __init__(self, prob=0.5, max_shear=0.25, center_offset=False):
        self._prob = prob
        self._max_shear = max_shear
        self._center_offset = center_offset

    def __call__(self, img):
        if np.random.rand() > self._prob:
            return img

        h, w = img.shape[:2]

        center_h = int((np.random.rand()*2-1) * h//4) if self._center_offset else 0
        center_w = int((np.random.rand()*2-1) * w//4) if self._center_offset else 0
        shear = np.random.rand() * self._max_shear

        data = img[..., :-1]
        target = img[..., -1]
        min_intensity = data.min()
        rng_intensity = data.max() - min_intensity

        matrix = np.array([[1, shear, -center_w], [shear, 1, -center_h], [0, 0, 1]])
        sheared_image = np.uint8(warp(data, matrix, order=3) * rng_intensity + min_intensity)
        sheared_target = np.uint8(warp(target, matrix, order=0) * 255.0)

        return np.concatenate((sheared_image, sheared_target[..., np.newaxis]), axis=2)


def AngioTransform(mode='train'):
    if 'train' in mode:
        transform = transforms.Compose([
            RandFlip(0.9, 0.5),
            RandomRotation(0.3, True),
            RandomShear(0.1, 0.2, True),
            RandGamma(0.5, 0.1, 1.9),
            transforms.ToTensor(),
            transforms.Pad((2, 2, 2, 2))
        ])
    elif 'valid' in mode or 'test' in mode:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad((2, 2, 2, 2))
        ])
    return transform


def ang_patches_collate(batch, patch_size=32, patch_stride=32, min_labeled_area=50, max_batch_size=64):
    batch_data = []
    batch_target = []
    for data, target in batch:
        batch_data.append(data)
        batch_target.append(target.unsqueeze(dim=0))
    batch_data = torch.cat(batch_data, dim=0)
    batch_target = torch.cat(batch_target, dim=0)

    batch_data = batch_data.unfold(2, patch_size, patch_stride).unfold(1, patch_size, patch_stride).reshape(-1, 1, patch_size, patch_size)
    batch_target = batch_target.unfold(2, patch_size, patch_stride).unfold(1, patch_size, patch_stride).reshape(-1, patch_size, patch_size)

    labeled_patches = list(filter(lambda i: batch_target[i].sum() > min_labeled_area, range(batch_target.size(0))))
    labeled_patches = np.random.choice(labeled_patches, min(max_batch_size, len(labeled_patches)), replace=False)
    batch_data = batch_data[labeled_patches, ...]
    batch_target = batch_target[labeled_patches, ...]

    return batch_data, batch_target


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from functools import partial

    root = r'D:\Test_data\Angios_134'
    ang_ds = AngioDB(root, mode='train', transform=AngioTransform('train'), size=-1)
    ang_dl = DataLoader(ang_ds, shuffle=True, batch_size=1, collate_fn=partial(ang_patches_collate, patch_size=32, patch_stride=32, min_labeled_area=50, max_batch_size=64))

    for i, (img, gt) in enumerate(ang_dl):
        print(i, img.shape, gt.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(img[0, 0], 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(gt[0], 'gray')
        plt.show()
