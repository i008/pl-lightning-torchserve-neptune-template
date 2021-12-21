import albumentations as A
from albumentations.pytorch import ToTensorV2

AUGMENTATIONS = {
    "hard_1": [
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),

    ],

    "easy": [A.Flip()]

}


def get_augment(augment_level):
    if augment_level not in AUGMENTATIONS:
        raise ValueError(f"Augmentation strategy has to be one of {AUGMENTATIONS.keys()}")
    return AUGMENTATIONS[augment_level]


def build_post_transform(normalization: str):
    if normalization == 'imagenet':
        post_transform = [A.Normalize(), ToTensorV2()]

    elif normalization == 'vit':
        post_transform = [A.Normalize(mean=(0.5, 0.5, 0.5),
                                      std=(0.5, 0.5, 0.5)), ToTensorV2()]

    else:
        raise ValueError("Wrong normalization type")

    return post_transform


def build_inference_transform(normalization, size=224):
    pre_transform = [A.LongestMaxSize(size), A.PadIfNeeded(size, size, border_mode=0)]
    post_transform = build_post_transform(normalization)
    return A.Compose([*pre_transform, *post_transform])


def build_training_transform(size, normalization, augment_level: str):
    pre_transform = [A.LongestMaxSize(size), A.PadIfNeeded(size, size, border_mode=0)]
    post_transform = build_post_transform(normalization)
    augment_transform = get_augment(augment_level)

    return A.Compose([*pre_transform, *augment_transform, *post_transform])


def build_eval_transform(normalization, size):
    pre_transform = [A.LongestMaxSize(size), A.PadIfNeeded(size, size, border_mode=0)]
    post_transform = build_post_transform(normalization)

    return A.Compose([*pre_transform, *post_transform])


def post_process_handle(data):
    """
    What happens wit the predictions of the model on torchserve side
    """
    return [{'logits': data['logits'].argmax(axis=1).tolist()}]
