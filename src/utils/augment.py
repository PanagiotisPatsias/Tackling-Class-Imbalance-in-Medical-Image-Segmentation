import albumentations as A


def default_aug():
    return A.Compose([
    A.VerticalFlip(p=0.15),
    A.HorizontalFlip(p=0.15),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.15, 0.25), rotate_limit=15, p=0.15),
    A.ElasticTransform(alpha=900, sigma=11, p=0.15),
    A.RandomBrightnessContrast(brightness_limit=(-0.5, 1.0), contrast_limit=0, p=0.15),
    ])