import os
import flwr as fl
from pytz import utc
import tensorflow as tf

import utils as uts

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #verbose

# Load model and data (MobileNetV2, CIFAR-10)
model = uts.new_model(uts.model) #def model
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"]) #compile
(x_train, y_train), (x_test, y_test) = uts.load_mydata() #read data

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters) #weights
        model.fit(x_train, y_train, epochs=1, batch_size=32) #fit
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters) #weights
        loss, accuracy = model.evaluate(x_test, y_test) #evaluate
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address=uts.sv, client=CifarClient())
