from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import numpy as np
from torch.nn import BCELoss
from torch.optim import Adam
import cv2
import numpy as np

NOISE_DIM = 300
MINIBATCH_SIZE = 128
IMG_SHAPE = (64, 64)
loss_fn = BCELoss()

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.initial_shape = (128, 60, 60)

        self.linear = nn.Linear(input_size, self.initial_shape[0] * self.initial_shape[1] * self.initial_shape[2])
        self.t_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=1)
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.t_conv2 = nn.ConvTranspose2d(64, 3, kernel_size=(3, 3), stride=1)
        self.batchnorm2 = nn.BatchNorm2d(3)

        # self.t_conv3 = nn.ConvTranspose2d(128, 3, kernel_size=(3,3), stride=1)
        # self.batchnorm3 = nn.BatchNorm2d(3)

        self.sigmoid = nn.Tanh()

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = x.reshape((x.shape[0], *self.initial_shape))

        x = self.t_conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.t_conv2(x)

        x = self.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3,3))
        self.relu = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(64, 128, (3,3))

        self.max_pool = nn.MaxPool2d((3,3))

        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(20*20*128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)

        x = self.max_pool(x)
        x = self.relu(x)

        x = self.flatten(x)
        return self.mlp(x)

def load_data():
    training_data = []
    print('loading data')
    for i in range(12500):
        try:
            img = np.array(cv2.imread('Cat/{}.jpg'.format(i)))
            img = cv2.resize(img, (IMG_SHAPE))
            img = (img.reshape((3, *IMG_SHAPE)) - 128.0) / 128.0
            img = torch.Tensor(img)
            # if (img.shape != (3, 28, 28)):
            #     print(img.shape)
            training_data.append(img)
        except:
            pass
    print('data loaded')
    return torch.stack(training_data)

def sample_real_img(dataset):
    imgs = []
    for _ in range(MINIBATCH_SIZE // 2):
        imgs.append(dataset[np.random.randint(0, len(dataset))])
    return torch.stack(imgs)

def sample_fake_img(generator):
    noises = torch.normal(0, 1, (MINIBATCH_SIZE // 2, NOISE_DIM))
    return generator(noises)

def show_img(img):
    cv2.imshow('img', cv2.resize(img.reshape((IMG_SHAPE))), (500, 500))
    cv2.waitKey(1)
    input('PRESS ANY KEY')

def get_acc(y_pred, y):
    num = 0
    for i in range(len(y)):
        if torch.round(y_pred[i][0]) == y[i][0]:
            num += 1
    return num / len(y)


def train_disc(dataset, generator, discriminator, d_opt, episode):
    losses = []
    for e in range(episode):
        generator.eval()
        discriminator.train()
        X = []
        y = []

        real_imgs = sample_real_img(dataset)
        X.append(real_imgs)

        # print(real_imgs.shape)

        imgs = sample_fake_img(generator)
        # print(imgs.shape)
        X.append(imgs.detach())
        X = torch.cat(X)

        y.append(torch.ones((MINIBATCH_SIZE//2, 1)))
        y.append(torch.zeros((MINIBATCH_SIZE//2, 1)))
        y = torch.cat(y)

        d_opt.zero_grad()

        outputs = discriminator(X)

        loss = loss_fn(outputs, y)
        loss.backward()

        d_opt.step()

        print('Discriminator loss: {}, acc: {}, num episode: {}'.format(loss.item(), get_acc(outputs, y), e))
        losses.append(loss.item())
    return losses

def train_gen(generator, discriminator, g_opt, episode):
    losses = []
    for e in range(episode):
        generator.train()
        discriminator.eval()
        X = torch.normal(0, 1, (MINIBATCH_SIZE, NOISE_DIM))
        y = torch.ones((MINIBATCH_SIZE, 1))

        g_opt.zero_grad()

        imgs = generator(X)

        outputs = discriminator(imgs)

        loss = loss_fn(outputs, y)

        loss.backward()

        g_opt.step()
        print('Generator loss: {}, acc: {}, num episode: {}'.format(loss.item(), get_acc(outputs, y), e))
        losses.append(loss.item())
    return losses

def save_model(nn, name):
    torch.save(nn.state_dict(), name)

def load_model(nn, name):
    nn.load_state_dict(torch.load(name))


if __name__ == '__main__':
    dataset = load_data()
    print(dataset.shape)

    disc = Discriminator()
    gen = Generator(NOISE_DIM)

    d_opt = Adam(disc.parameters(), lr=0.0002)
    g_opt = Adam(gen.parameters(), lr=0.0002)

    generator_loss = []
    discriminator_loss = []

    for _ in range(100000000000):
        discriminator_loss += train_disc(dataset, gen, disc, d_opt, 10)
        generator_loss += train_gen(gen, disc, g_opt, 10)

        save_model(gen, 'gen')
        save_model(disc, 'disc')

        np.save('gloss.npy', generator_loss)
        np.save('dloss.npy', discriminator_loss)

