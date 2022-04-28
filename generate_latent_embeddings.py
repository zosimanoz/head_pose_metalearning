import os
import torch
import random
import utils
from dataloaders.BiwiDataLoader import BiwiDataLoader
from models.vae import VAE
import numpy as np
from torch.autograd import Variable


RANDOM_SEED = 123
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)
    
def set_global_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_data(filename):
    dataset = utils.load_npz(filename)
    data = BiwiDataLoader(dataset)
    return data

def get_embeddings(model, dataloader):
    all_embeddings = []
    for images, labels, folder, path in dataloader:
        images = np.array(images) / 255
        images = np.moveaxis(images, 3, 1)
        images = torch.from_numpy(images)
        images = Variable(images).float()
        images = images.to(DEVICE)

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        path = np.array(path)

        with torch.no_grad():
            embeddings = model.get_encodings(images).to('cpu')

        for i in range(len(embeddings)):
            all_embeddings.append([embeddings[i].cpu().numpy(), labels[i].cpu().numpy(), path[i]])

    return all_embeddings


def main():
    filename = 'datasets/biwi_dataset_128_full.npz'
    data = get_data(filename)

    # create train_dataloader
    train_loader = torch.utils.data.DataLoader(dataset=data.ds_train,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=0
                                               )

    # load checkpoint with best VAE model state
    model = VAE()
    model.load_state_dict(torch.load('checkpoints/embeddings/vae_maml_final_new.pt', map_location=torch.device('cpu')), strict=False)
    model.to(DEVICE)

    # create embeddings to train meta learner
    all_embeddings = get_embeddings(model, train_loader)

    # save embeddings
    np.savez('latent_embeddings/embeddings_full.npz', all_embeddings)

    print('Finished saving latent embeddings !')


if __name__ == '__main__':
    main()

