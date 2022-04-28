import numpy as np

class BiwiDataLoader():
    def __init__(self, dataset):
        self.dataset = dataset
        self.ds_train = []
        self.train = []

        for img, pose, folder, path in zip(self.dataset['image'], self.dataset['pose'], self.dataset['folder_name'], self.dataset['image_name']):
            self.ds_train.append([img, pose, folder, path])

    def __getitem__(self, index):
        img, pose, folder, path = self.ds_train[index]
        return img, pose, folder, path, self.ds_train[index]

    def __len__(self):
        return len(self.ds_train)