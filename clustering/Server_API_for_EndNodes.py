# Import the framework
from flask import Flask, g
from flask_restful import Resource, Api, reqparse
from IoT_System import DeviceList
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils

# Create an instance of Flask
app = Flask(__name__)
# Create the API
api = Api(app)
End_Nodes = DeviceList()
# ==================================== Load DQN and RNN model from json file====================================#
DRL_json_file = open('DRL_models/DRL_model_Final_augmented.json', 'r')
loaded_model_json = DRL_json_file.read()
DRL_json_file.close()
DRL_model = model_from_json(loaded_model_json)
# load weights into new model
DRL_model.load_weights("DRL_models/DRL_model_Final_augmented.h5")
print("Loaded DRL model from disk")

RNN_json_file = open('LSTM_models/LSTM_full_data', 'r')
loaded_model_json_ = RNN_json_file.read()
RNN_json_file.close()
RNN_model = model_from_json(loaded_model_json_)
# load weights into new model
RNN_model.load_weights("LSTM_models/LSTM_full_data.h5")
print("Loaded RNN model from disk")

# time steps of LSTM mdoel
time_steps = 60
# Whole server number in the IoT network
server_number = 4
# Actions of the the DQN model.
model_action_number = 7
total_behavior_dim = 9
model_output_size = server_number * model_action_number

class Device_configuration(Resource):
    def get(self, identifier, network_state, truck_history_positions):
        # If the key does not exist in the data store, return a 404 error.
        if not (identifier in End_Nodes.devices()):
            return {'message': 'Device not found', 'data': {}}, 404

        truck_behavior_prediction = RNN_model.predict(
            np.reshape(truck_history_positions, (-1, time_steps, 2)), batch_size=1)  # Here implementing cross-probability
        q_table = np.zeros((1, model_output_size))
        for i in range(total_behavior_dim):
            behavior_array = np_utils.to_categorical(i, num_classes=total_behavior_dim)
            network_state = np.asarray(list(network_state) + list(behavior_array))
            q_table += DRL_model.predict(network_state.reshape(1, -1), batch_size=1) * float(truck_behavior_prediction[0, i])
        action = np.argmax(q_table[0])
        return {'message': 'Device found, configuration action sent', 'configuration action': action}, 200

api.add_resource(Device_configuration, '/devices')