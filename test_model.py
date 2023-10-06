from train import *


net = Generator(300)
load_model(net, 'gen')

noise = torch.normal(0, 1, (20, 300))
img = net(noise)

# dataset = load_data()

# data = dataset[np.random.randint(0,len(dataset))].numpy()
# print(data.shape)
#cv2.imwrite('gen_img/orig_sample.jpg', cv2.resize((data.reshape(64, 64, 3)*128 + 128).astype(np.uint8), (150, 150)))
for i in range(len(img)):
    cv2.imwrite('gen_img/gen_img_{}.jpg'.format(i), cv2.resize((img[i].detach().numpy().reshape((64,64,3))*128+128).astype(np.uint8), (150, 150)))
# show_img(img[0].detach().numpy().reshape(28, 28))
