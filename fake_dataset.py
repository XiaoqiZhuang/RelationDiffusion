import torch 
from PIL import Image
import torchvision.transforms as transforms


class FIGDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, caption, image_size=512, normalize=True):
        self.image_paths = image_paths
        self.image_size = image_size
        self.caption = caption

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),                      
            transforms.Normalize((0.5,), (0.5,)) if normalize else transforms.Lambda(lambda x: x) 
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, self.caption
