
from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
# from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import sys


class GAN():
    def __init__(self):
        self.data_shape = 6 # (28, 28, 1)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

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
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
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
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        data = Input(shape=self.data_shape)
        validity = model(data)

        return Model(data, validity)
    
    def load_predata(self):
        global nongen_data
        seed = 7
        np.random.seed(seed)
        # 載入141-7的訓練和測試資料集
        df_train = pd.read_csv("./141-7_train.csv").values
        # 打亂資料
        np.random.shuffle(df_train)
        # minority 的 label
        minor = np.array(['AppleiCloud', 'AppleiTunes', 'FTP_DATA', 'NetFlix',
                          'Snapchat', 'SoundCloud', 'Steam', 'TeamViwer',
                          'Telegram', 'Wikipedia', 'eBay'])
        # 取出 minority
        mino_data = np.empty(11)

        for sample in df_train:
            for mr in minor:
                if sample[10] == mr:
                    mino_data = np.vstack((mino_data, sample))
                    
        mino_data = np.delete(mino_data, 0, 0)
        nongen_data = pd.DataFrame(mino_data[:, 0:4])
        mino_data = np.delete(mino_data, [0,1,2,3,10], 1)
        # MinMax標準化
        train = pd.DataFrame(
            MinMaxScaler(feature_range=(0, 1)).fit(mino_data).transform(mino_data))
        
        return train

    
    def train(self, epochs, batch_size=128, sample_interval=50):
        global train_gen
        # Load the dataset
        train_gen = self.load_predata()

        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, train_gen.shape[0], batch_size)
            data = train_gen.values[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_data = self.generator.predict(noise)

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

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_data(epoch)

    def sample_data(self, epoch):
        global gen_data
        
        num = 7300 * 2 # 生成數
        noise = np.random.normal(0, 1, (num, self.latent_dim))
        gen_data = pd.DataFrame(self.generator.predict(noise))

        # # Rescale images 0 - 1
        # gen_data = 0.5 * gen_data + 0.5


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, sample_interval=30000) # 總生成次數, 單次生成數, 每生成n印出
    
    # 儲存生成資料
    # nongen_data = pd.concat([nongen_data, nongen_data], axis=0).reset_index(drop=True)
    # Gen_Data = pd.concat([nongen_data, gen_data], axis=1)
    # Gen_Data.to_csv('./data-gen.csv', index=False)
    gen_data.to_csv('./data-gen.csv', index=False)
    