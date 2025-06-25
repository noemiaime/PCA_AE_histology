# si è cercato di costruire una funzione abbastanza versatile, in modo da poterla utilizzare 
# su dataset con necessità differentisull'ulteriore dtataset, trattando il progetto di gruppo dataset differenti
def augment_data(
    samples, # lista di tuple (path, label)
    class_names, # nomi classi
    image_size=224, # dimensioni immagini
    grayscale=False, # scala colori
    mode="balance", # modalità "balance" per bilanciare il dataset
    augment_factor=2,# augment factor per modalità 'augment'
    add_noise=False, # aggiunta noise oppure no
    noise_mean=0.0, # parametri noise
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

    # trasformazioni PIL
    pil_augment = transforms.Compose([
        transforms.Resize((image_size, image_size)), # ridimensionamento
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(), # flip orizzontale
            transforms.RandomVerticalFlip(), # flip verticale
            transforms.RandomRotation(180),  # rotazione random fino a +- 180 gradi
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05), # variazione luminosità, contrasto, saturazione
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), # trasfromazioni geometriche, come rotazione, traslazione, zoom e inclinazione
            transforms.RandomPerspective(distortion_scale=0.5, p=1.0), # prospettiva
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0), # nitidezza
        ], p=0.7) # applicazione random trasformazioni con probabilità 0.7
    ])

    # trasformazioni appplicabili post trasformazione in tensore
    tensor_ops = [
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)), # sfocatura gaussiana
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random') # cancellazione random
    ]
    if add_noise:
        tensor_ops.append(AddGaussianNoise(mean=noise_mean, std=noise_std)) # rumore gaussiano se add_noise = True

    tensor_augment = transforms.RandomApply(tensor_ops, p=0.7) # applica tensor_ops con una probabilità p = 0.7
    to_tensor = transforms.ToTensor() # trasformazione in tensore

    # organizzazioni immagini per classe
    class_to_paths = {}
    for path, label in samples:
        class_to_paths.setdefault(label, []).append(path)

    if mode == 'balance': # mode 'balance' classi uguali alla più numerosa
        target_per_class = max(len(paths) for paths in class_to_paths.values())
    else:  # mode 'augment' ogni classe è moltiplicata per un fattore+
        target_per_class = max(int(len(paths) * augment_factor) for paths in class_to_paths.values())

    augmented_images, augmented_labels = [], []

    for label, paths in class_to_paths.items(): # itera su ogni classe
        orig_len = len(paths)
        extra_needed = target_per_class - orig_len # immagini extra da aggiungere

        # immagini originali
        for p in paths:
            img = Image.open(p).convert('L' if grayscale else 'RGB') # apertura immagine 
            img = pil_augment(img) # traformazioni pil_augment
            tensor_img = to_tensor(img) # traformazione in tensore
            tensor_img = tensor_augment(tensor_img) # trasformazioni su tensore
            augmented_images.append(tensor_img) # salvataggio
            augmented_labels.append(label)

        # immagini aggiunte
        for _ in range(extra_needed):
            p = random.choice(paths) # duplicazione di alcune random 
            img = Image.open(p).convert('L' if grayscale else 'RGB') # apertura immagine 
            img = pil_augment(img)  # traformazioni pil_augment
            tensor_img = to_tensor(img) # traformazione in tensore
            tensor_img = tensor_augment(tensor_img) # trasformazioni su tensore
            augmented_images.append(tensor_img)  # salvataggio
            augmented_labels.append(label)

    images_tensor = torch.stack(augmented_images) 

    return images_tensor, torch.tensor(augmented_labels), class_names # restituisce tensori di immagini, label e class_names 
