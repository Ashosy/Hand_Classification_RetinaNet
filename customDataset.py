import torch

class CustomDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self, tensors, transforms):
        super(CustomDataset, self).__init__()
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]
        



        if self.transforms:
            image = self.transforms(image)

        return (image, label, bbox)



