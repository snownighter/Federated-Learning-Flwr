import os
import flwr as fl

import tensorflow as tf

# add package
import utils as uts
# free memory
import reset_keras as reset
# ignore warn
import warnings
warnings.filterwarnings('ignore')

cl_num = 3 # client-number
round = 1 # global-round

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #verbose

# Load model and data (MobileNetV2, CIFAR-10)
(x_train, y_train), (x_test, y_test) = uts.load_data(num=3) #read data
model = uts.new_model(name=uts.model, input_size=x_train.shape[1]) #def model
model.compile(loss="categorical_crossentropy", optimizer=uts.adam, metrics=["accuracy"]) #compile

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        # Update local model parameters
        model.set_weights(parameters) #weights
        reset.reset_keras()
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=uts.mp.epochs , batch_size=uts.mp.batch_size) #fit
        if uts.drimg:
            global round
            uts.draw_img(history, cl_num, round); round += 1
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        #model.set_weights(parameters) #weights
        reset.reset_keras()
        loss, accuracy = model.evaluate(x_train, y_train) #evaluate train
        loss, accuracy = model.evaluate(x_test, y_test) #evaluate test
        if uts.wres:
            uts.result(model, cl_num, round-1)
        # Update local model with global parameters
        model.set_weights(parameters)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address=uts.sv, client=CifarClient())
