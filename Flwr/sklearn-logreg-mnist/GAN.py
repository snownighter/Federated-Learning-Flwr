
from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.utils import resample, shuffle

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np

import time

class GAN():
    def __init__(self):
        
        self.minor = ['FTP_DATA', 'Snapchat', 'SoundCloud', 'eBay'] # 微流量 之 name
        self.support = [52, 57, 42, 110] # 微流量 之 support
        self.rate = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0] # 生成比例
        
        self.data_shape = 6
        self.latent_dim = 100

        # 1: SGD(0.0002, 0.7) batch_size=64
        # 2: optimizer = Adam(0.0002, 0.7)
        optimizer = Adam(0.0002, beta_1=0.7, decay=0.0005) # 3: re-MLP2

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates data
        z = Input(shape=(self.latent_dim,))
        data = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(data)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512)) # 512
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.data_shape, activation='sigmoid'))
        # model.add(Reshape(self.data_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        data = model(noise)

        return Model(noise, data)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.data_shape, activation="relu"))
        model.add(Dense(512)) # 512
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        data = Input(shape=self.data_shape)
        validity = model(data)

        return Model(data, validity)
    
    def load_predata(self, index):
        
        global nongen, gen, label
        
        data = pd.read_csv("./data/minmax-train.csv")
        
        str_name = ['src_ip', 'dst_ip', 'server_port', 'prot']
        num_name = ['p_count', 'b_count', 'max_size', 'min_size', 'abyte_count', 'sbyte_count']
        lab_name = ['label']
        
        data.columns = str_name + num_name + lab_name
        
        
        mino = data.query('label==@self.minor[@index]')
        nongen, gen, label = mino.loc[:, str_name], mino.loc[:, num_name], mino.loc[:, lab_name]
        
        return gen

    
    def train(self, epochs, batch_size=128, index=0):
        
        global nongen, gen, label
        # Load the dataset
        gen = self.load_predata(index)
        
        # 儲存loss
        dloss, gloss = [], []

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, gen.shape[0], batch_size)
            data = gen.values[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_data = self.generator.predict(noise, verbose=0)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            dloss.append(d_loss[0])
            gloss.append(g_loss)

            if epoch % 1000 == 0:
                # Plot the progress
                print ("%d %d- [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, index+1, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch == epochs-1:
                
                num_name = ['p_count', 'b_count', 'max_size', 'min_size', 'abyte_count', 'sbyte_count']
        
                data = []
                for i in range(len(self.rate)):
                    num = int(self.support[index] * self.rate[i]) # 生成數
                    noise = np.random.normal(0, 1, (num, self.latent_dim))
                    pred = pd.DataFrame(self.generator.predict(noise), columns=num_name)
                    
                    nonpred = resample(nongen, n_samples=num, replace=True, stratify=nongen)
                    nonpred = nonpred.reset_index(drop=True)
                    
                    lab = resample(label, n_samples=num, replace=True, stratify=label)
                    lab = lab.reset_index(drop=True)
                    lab.columns = ['']
                    
                    data.append(pd.concat([nonpred, pred, lab], axis=1))
                
                # 顯示訓練損失
                epochs = range(1, len(gloss)+1)
                plt.figure(figsize=(15, 12))
                plt.plot(epochs, gloss, linewidth=0.3, label="G Loss") # , "b-"
                plt.plot(epochs, dloss, linewidth=0.3, label="D Loss") # , "r-"
                plt.title("Generator and Discriminator Loss (" + self.minor[index] + ")", fontsize=25)
                plt.xlabel("Epochs", fontsize=16)
                plt.ylabel("Loss", fontsize=16)
                plt.gca().xaxis.set_major_locator(MultipleLocator(1000))
                plt.gca().yaxis.set_major_locator(MultipleLocator(.5))
                plt.ylim(0,3)
                plt.legend()
                plt.savefig('./Loss_img/Loss-for-' + self.minor[index] + '.png')
                plt.close()
                
                # 儲存loss值
                dloss = pd.DataFrame(dloss, columns=['d_loss'])
                gloss = pd.DataFrame(gloss, columns=['g_loss'])
                pd.concat([gloss, dloss], axis=1).to_csv('./data/loss(' + self.minor[index] + ').csv', index=False)
        
                return data

start_time = time.ctime(time.time()) # 開始

if __name__ == '__main__':
    g = GAN()
    def Gan(i):
        gan = GAN() # gan.append(GAN())
        return gan.train(epochs=8000, batch_size=64, index=i) # 1/2: 10000, 64
    
    def save_data():
        data1, data2, data3, data4 = Gan(0), Gan(1), Gan(2), Gan(3)
        for i in range(len(g.rate)):
            data = pd.concat([data1[i], data2[i], data3[i], data4[i]], axis=0)
            data = data.reset_index(drop=True)
            
            data.to_csv('./data/data-gen(rate=' + str(g.rate[i]) + ').csv', index=False)
    
    save_data()
    
end_time = time.ctime(time.time()) # 結束
print(start_time + '\n' + end_time)