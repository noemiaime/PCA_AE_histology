def build_balanced_augmented_tensor_dataset(
    samples, 
    class_names,
    image_size=224,
    grayscale=False,
    mode="balance",
    augment_factor=2,
    add_noise=False,
    noise_mean=0.0,
    noise_std=0.05
):
    from torchvision import transforms
    from PIL import Image
    import torch
    import random

    class AddGaussianNoise(object):
        def __init__(self, mean=0.0, std=0.05):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)

    # --- Augmentazioni PIL ---
    pil_augment = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
        ], p=0.7)
    ])

    tensor_ops = [
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
    ]
    if add_noise:
        tensor_ops.append(AddGaussianNoise(mean=noise_mean, std=noise_std))

    tensor_augment = transforms.RandomApply(tensor_ops, p=0.7)
    to_tensor = transforms.ToTensor()

    # --- Organizzazione immagini per classe ---
    class_to_paths = {}
    for path, label in samples:
        class_to_paths.setdefault(label, []).append(path)

    if mode == 'balance':
        target_per_class = max(len(paths) for paths in class_to_paths.values())
    else:  # mode == 'augment'
        target_per_class = max(int(len(paths) * augment_factor) for paths in class_to_paths.values())

    augmented_images, augmented_labels = [], []

    for label, paths in class_to_paths.items():
        orig_len = len(paths)
        extra_needed = target_per_class - orig_len

        # Originali
        for p in paths:
            img = Image.open(p).convert('L' if grayscale else 'RGB')
            img = pil_augment(img)
            tensor_img = to_tensor(img)
            tensor_img = tensor_augment(tensor_img)
            augmented_images.append(tensor_img)
            augmented_labels.append(label)

        # Augmentate
        for _ in range(extra_needed):
            p = random.choice(paths)
            img = Image.open(p).convert('L' if grayscale else 'RGB')
            img = pil_augment(img)
            tensor_img = to_tensor(img)
            tensor_img = tensor_augment(tensor_img)
            augmented_images.append(tensor_img)
            augmented_labels.append(label)

    images_tensor = torch.stack(augmented_images)

    return images_tensor, torch.tensor(augmented_labels), class_names
