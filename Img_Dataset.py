from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, pathNoise, pathGT):
        super(ImageDataset, self).__init__()
        self.noise_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.gt_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.filesNoise = sorted(glob.glob(pathNoise + "/*.*"))
        self.filesGT = sorted(glob.glob(pathGT + "/*.*"))

    def __getitem__(self, index):
        imgNoise = Image.open(self.filesNoise[index % len(self.filesNoise)])
        imgGT = Image.open(self.filesGT[index % len(self.filesGT)])

        img_noise = self.noise_transform(imgNoise)
        img_gt = self.gt_transform(imgGT)

        return {'noise': img_noise, 'GT': img_gt}

    def __len__(self):
        return len(self.filesGT)