import numpy as np
from keras.models import model_from_json

while 1:
    input  = np.random.randint(5, size=(6562,) )
    print(input)
    DRL_json_file = open('DRL models/DRL_model_full_training_data_tackling_overfitting.json', 'r')
    loaded_model_json = DRL_json_file.read()
    DRL_json_file.close()
    DRL_model = model_from_json(loaded_model_json)
    # load weights into new model
    DRL_model.load_weights("DRL models/DRL_model_full_training_data_tackling_overfitting.h5")
    q_table = DRL_model.predict(np.asarray(input).reshape(1, -1), batch_size=1)
    print(q_table)