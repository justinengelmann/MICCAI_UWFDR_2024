import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF

import numpy as np
from scipy.stats import gamma
import cv2
from PIL import Image
from torchvision.transforms import functional as TF

def add_horizontal_stripe_artifact(
    image, 
    max_halfwidth=12, 
    min_halfwidth=0, 
    max_intensity=0.8, 
    min_intensity=0.1,
    max_red_channel_factor=0.95, 
    min_red_channel_factor=0.4,
    b_channel_factor=0, 
    mean_blend_fact=0.7, 
    min_mean_brighten_fact=0.2,
    max_mean_brighten_fact=1.4,
    is_rgb=True
):
    def calculate_artifact_dimensions():
        half_width = np.random.randint(min_halfwidth, max_halfwidth + 1)
        center_row = np.random.randint(0, height)
        start_row = max(0, center_row - half_width)
        end_row = min(height, center_row + half_width + 1)
        return half_width, start_row, end_row

    def calculate_intensity(half_width):
        width_moderation_fact = 1 - (half_width / (max_halfwidth * 2))
        intensity = np.random.uniform(min_intensity, max_intensity) * width_moderation_fact
        return max(1.05, intensity + 1)

    def generate_weights(artifact_height, max_intensity):
        return np.concatenate([
            np.linspace(1., max_intensity, artifact_height // 2 + 1),
            np.linspace(max_intensity, 1, artifact_height - artifact_height // 2)[1:]
        ])

    def generate_speckle(artifact_height, width):
        shape, scale = 4, 0.02
        speckle = gamma.rvs(shape, scale=scale, size=(artifact_height, width))
        speckle = np.clip(speckle, 0.03, None) + 1
        speckle += gamma.rvs(shape, scale=scale, size=(artifact_height, width))
        return speckle

    def apply_artifact(artifact_region, intensity_map, channel, factor, mean_brighten_fact):
        artifact_region[:, :, channel] *= (1 + (intensity_map - 1) * factor)
        artifact_region[:, :, channel] = (
            means[channel] * mean_brighten_fact * mean_blend_fact + 
            (1 - mean_blend_fact) * artifact_region[:, :, channel]
        )

    height, width, _ = image.shape
    half_width, start_row, end_row = calculate_artifact_dimensions()
    artifact_height = end_row - start_row
    
    max_intensity = calculate_intensity(half_width)
    r_channel_factor = np.random.uniform(min_red_channel_factor, max_red_channel_factor)
    
    weights = generate_weights(artifact_height, max_intensity)
    speckle = generate_speckle(artifact_height, width)
    intensity_map = weights[:, np.newaxis] * speckle
    
    artifact_region = image[start_row:end_row, :, :]
    green_channel, red_channel, blue_channel = (1, 0, 2) if is_rgb else (1, 2, 0)
    
    means = artifact_region.mean(axis=(0,1))
    width_moderation_fact = 1 - (half_width / (max_halfwidth * 2))
    green_mean_brighten_fact = np.random.uniform(min_mean_brighten_fact, max_mean_brighten_fact) * width_moderation_fact + 1
    red_mean_brighten_fact = max(1.05, r_channel_factor * green_mean_brighten_fact)

    green_mean_brighten_fact = np.random.normal(loc=green_mean_brighten_fact, scale=0.15, size=artifact_region.shape[:-1])
    red_mean_brighten_fact = np.random.normal(loc=green_mean_brighten_fact, scale=0.15, size=artifact_region.shape[:-1])
    
    apply_artifact(artifact_region, intensity_map, green_channel, 1, green_mean_brighten_fact)
    apply_artifact(artifact_region, intensity_map, red_channel, r_channel_factor, red_mean_brighten_fact)
    
    if b_channel_factor > 0:
        apply_artifact(artifact_region, intensity_map, blue_channel, b_channel_factor, 1)
    
    return image.clip(0, 1)

def add_multiple_horizontal_stripe_artifacts(image):
    n_thin_strong_stripes = np.random.randint(2, 15)
    n_thick_weak_stripes = np.random.randint(2, 15)
    image_with_artifact = image.copy().astype(float) / 255.
    for _ in range(n_thin_strong_stripes):
        image_with_artifact = add_horizontal_stripe_artifact(image_with_artifact, is_rgb=True,
                                                             min_halfwidth=0, max_halfwidth=2, 
                                                             min_mean_brighten_fact=0.6, max_mean_brighten_fact=1.8,
                                                             min_intensity=0.2, max_intensity=1.2,
                                                             mean_blend_fact=0.8)
    for _ in range(n_thick_weak_stripes):
        image_with_artifact = add_horizontal_stripe_artifact(image_with_artifact, is_rgb=True,
                                                             min_halfwidth=3, max_halfwidth=20, max_intensity=0.5, max_mean_brighten_fact=0.3, mean_blend_fact=0.3)

    image_with_artifact *= 255.
    return image_with_artifact.astype(np.uint8)

def make_blurry(image):
    scale_fact = np.random.uniform(0.06, 0.18)
    image = np.array(image).astype(np.uint8)
    size = image.shape[:-1]
    lowsize = (int(size[0]*scale_fact), int(size[1]*scale_fact))
    image = Image.fromarray(image.astype(np.uint8))
    image = TF.resize(image, size=lowsize)
    image = TF.resize(image, size=size)
    return np.array(image)

def make_dark(image):
    bright_fact = np.random.uniform(0.4, 0.8)
    contrast_fact = np.random.uniform(0.4, 0.95)
    image = np.array(image).astype(np.uint8)
    image = Image.fromarray(image)
    image = TF.adjust_contrast(image, contrast_fact)
    image = TF.adjust_brightness(image, bright_fact)
    return np.array(image)

def make_bright(image):
    red_fact = np.random.uniform(0.95, 1.4)
    bright_fact = np.random.uniform(1.1, 1.5)
    contrast_fact = np.random.uniform(1., 1.3)
    gamma_fact = np.random.uniform(0.7, 2)

    if np.random.uniform(0,1) < 0.2:
        image = np.array(image).astype(float)
        image[:, :, 0] *= red_fact
    image = Image.fromarray(image.astype(np.uint8))
    image = TF.adjust_brightness(image, bright_fact)
    image = TF.adjust_gamma(image, gamma_fact, gain=contrast_fact)
    
    return np.array(image)

def make_red(image):
    red_fact = np.random.uniform(1.2, 2.2)
    image = np.array(image).astype(float)
    image[:, :, 0] *= red_fact
    image = Image.fromarray(image.astype(np.uint8))    
    return np.array(image)

def make_speckly(image):
    image = np.array(image).astype(float)
    image *= np.random.normal(loc=1., scale=0.05, size=image.shape)
    image = Image.fromarray(image.astype(np.uint8))    
    return np.array(image)


class CustomImageDatasetUWFTask1(Dataset):
    def __init__(self, df, img_dir, target_cols, resolution, transform=None):
        self.df = df
        self.df['make_bad'] = 0
        df_extra = df[df['image quality level']==1].copy()
        df_extra['make_bad'] = 1
        self.df = pd.concat([self.df, df_extra])
        self.img_dir = img_dir
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)
        self.transform = transform

    def __len__(self):
        return len(self.df) 
        
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Read and resize image
        image = cv2.imread(img_path, 1)

        # Get target(s) as float
        target = torch.tensor([float(self.df.iloc[idx][col]) for col in self.target_cols])

        if self.df.iloc[idx]['make_bad'] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
            # # Normalize the image to float32 with values between 0 and 1
            # image = image.astype(np.float32) / 255.0
            image_og = image.copy()

            
            blurred = np.random.uniform(0,1) < 0.2
            if blurred:
                image = make_blurry(image)

            speckly =  np.random.uniform(0,1) < 0.15
            if speckly:
                image = make_speckly(image)

            brightness = np.random.uniform(0,1) < 0.3
            if brightness:
                if np.random.uniform(0,1) < 0.2:
                    image = make_dark(image)
                else:
                    image = make_bright(image)
            else:
                if np.random.uniform(0,1) < 0.2:
                    image = make_red(image)

            stripes = np.random.uniform(0,1) < 0.8
            if stripes or not blurred:
                image = add_multiple_horizontal_stripe_artifacts(image)

            # Normalize the image to float32 with values between 0 and 1
            image = image.astype(np.float32) / 255.0
            image_og = image_og.astype(np.float32) / 255.0

            if np.random.uniform(0,1) < 0.9:
                x_left, x_right, y_top, y_bottom = np.random.uniform(0, 0.2, size=4)
    
                x_left, x_right = int(x_left * image_og.shape[1]), int(x_right * image_og.shape[1])
                y_top, y_bottom = int(y_top * image_og.shape[0]), int(y_bottom * image_og.shape[0])

                if np.random.uniform(0,1) < 0.8:
                    image[:, :x_left] = image_og[:, :x_left]
                if np.random.uniform(0,1) < 0.8:
                    image[:, -x_right:] = image_og[:, -x_right:]
                if np.random.uniform(0,1) < 0.8:
                    image[:y_top] = image_og[:y_top]
                if np.random.uniform(0,1) < 0.8:
                    image[-y_bottom:] = image_og[-y_bottom:]

            
            image = image * 255
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            target = torch.tensor([float(0.01)])

        image = cv2.resize(image, self.resolution)
        
        # Convert to PIL Image for torchvision T
        image = T.ToPILImage()(image)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'target': target}

class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir, target_cols, resolution, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Read and resize image
        image = cv2.imread(img_path, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.resolution)
        
        # Convert to PIL Image for torchvision T
        image = T.ToPILImage()(image)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Get target(s) as float
        target = torch.tensor([float(self.df.iloc[idx][col]) for col in self.target_cols])
        
        return {'image': image, 'target': target}

def get_augmentations(aug_type='default', aug_prob=0.99,
                      norm_mean=(0.5,), norm_std=(0.5,)):
    if aug_type == 'default':
        augs = [
            T.RandomAffine(degrees=12, scale=(0.93, 1.08)),
            T.RandomAffine(degrees=1, shear=5),
            T.ColorJitter(brightness=0.25, contrast=0.25),
            T.RandomApply([T.TrivialAugmentWide()], p=0.25),
        ]
        re_augs = [
            T.RandomErasing(p=0.15, scale=(0.03, 0.15)),          
            T.RandomErasing(p=0.15, scale=(0.03, 0.15)),          
            T.RandomErasing(p=0.15, scale=(0.03, 0.15)),          
            T.RandomErasing(p=0.15, scale=(0.03, 0.15)),          
        ]
        augs = T.Compose([T.RandomApply([_], p=aug_prob) for _ in augs])
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            augs,
            T.ToTensor(),
            T.Compose(re_augs),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    elif aug_type == 'strong':
        augs = [
            T.RandomAffine(degrees=18, scale=(0.8, 1.1)),
            T.RandomAffine(degrees=1, shear=8),
            T.ColorJitter(brightness=0.5, contrast=0.5),
            T.RandomApply([T.TrivialAugmentWide()], p=0.66),
        ]
        re_augs = [T.RandomErasing(p=0.2, scale=(0.01, 0.2)) for _ in range(10)]
        augs = T.Compose([T.RandomApply([_], p=aug_prob) for _ in augs])
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.25),
            augs,
            T.ToTensor(),
            T.Compose(re_augs),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    elif aug_type == 'strong_v2':
        augs = [
            T.RandomAffine(degrees=18, scale=(0.8, 1.1)),
            T.RandomAffine(degrees=1, shear=8),
            T.ColorJitter(brightness=0.25, contrast=0.15),
            T.RandomApply([T.TrivialAugmentWide()], p=0.66),
        ]
        re_augs = [T.RandomErasing(p=0.2, scale=(0.01, 0.2), value=0.75) for _ in range(12)]
        augs = T.Compose([T.RandomApply([_], p=aug_prob) for _ in augs])
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.25),
            augs,
            T.ToTensor(),
            T.Compose(re_augs),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    elif aug_type == 'quality':
        augs = [
            T.RandomAffine(degrees=12, scale=(0.93, 1.08)),
            # T.ColorJitter(brightness=0.05, contrast=0.05),
        ]
        augs = T.Compose([T.RandomApply([_], p=aug_prob) for _ in augs])
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            augs,
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    elif aug_type == 'quality_re':
        augs = [
            T.RandomAffine(degrees=5, scale=(0.93, 1.08)),
            # T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        re_augs = [T.RandomErasing(p=0.2, scale=(0.01, 0.1), value=0.75) for _ in range(10)]
        augs = T.Compose([T.RandomApply([_], p=aug_prob) for _ in augs])
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            augs,
            T.ToTensor(),
            T.Compose(re_augs),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    elif aug_type == 'quality_rev2':
        augs = [
            T.RandomAffine(degrees=5, scale=(0.93, 1.08)),
            T.ColorJitter(brightness=0.02, contrast=0.02),
        ]
        re_augs = [T.RandomErasing(p=0.2, scale=(0.01, 0.1), value=0.75) for _ in range(20)]
        augs = T.Compose([T.RandomApply([_], p=aug_prob) for _ in augs])
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            augs,
            T.ToTensor(),
            T.Compose(re_augs),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    elif aug_type == 'quality_rev3':
        augs = [
            T.RandomAffine(degrees=5, scale=(0.98, 1.02)),
            # T.ColorJitter(brightness=0.02, contrast=0.02),
        ]
        re_augs = [T.RandomErasing(p=0.2, scale=(0.01, 0.05), value=0.75) for _ in range(20)]
        augs = T.Compose([T.RandomApply([_], p=aug_prob) for _ in augs])
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.4),
            augs,
            T.ToTensor(),
            T.Compose(re_augs),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    elif aug_type == 'none':
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

def get_dataloaders(train_df, val_df, target_cols, res=768, img_dir='all_imgs_all/', aug_type='default', aug_prob=0.99, 
                    uwf_task1_extra_augs=False, 
                    batch_size=32, norm_mean=(0.5,), norm_std=(0.5,)):
    # Get augmentations
    train_transform = get_augmentations(aug_type, aug_prob, norm_mean, norm_std)
    val_transform = get_augmentations('none', norm_mean=norm_mean, norm_std=norm_std)
    
    # Create datasets
    train_dataset = CustomImageDataset(train_df, img_dir, target_cols, res, transform=train_transform)
    if uwf_task1_extra_augs:
        train_dataset = CustomImageDatasetUWFTask1(train_df, img_dir, target_cols, res, transform=train_transform)
    val_dataset = CustomImageDataset(val_df, img_dir, target_cols, res, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, val_loader

