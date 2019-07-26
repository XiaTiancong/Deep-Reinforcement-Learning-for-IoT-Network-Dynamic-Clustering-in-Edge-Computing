import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
from read_AGV_data import AGV

threshold = 0.3
sample_period = 0.1
grid_size = 0.025
categorial = True

agv = AGV('training', 0.25)
df = agv.get_df()
df = agv.fit_to_canvas(df, 15)
df_sampled = agv.identical_sample_rate(df, sample_period)
grided_pos = agv.trace_grided(df_sampled, grid_size)
trace_grided = agv.delete_staying_pos(grided_pos)


time_steps = 60
train_propotion = 0.8
train_set_num = int(round(len(trace_grided) * train_propotion))
train_set = trace_grided[:train_set_num]
test_set = trace_grided[train_set_num:]

train_input, train_output = agv.get_inputs_and_outputs(train_set, time_steps)
test_input, test_output = agv.get_inputs_and_outputs(test_set, time_steps)
train_dir_output = agv.get_dir(train_input, train_output)
test_dir_output = agv.get_dir(test_input, test_output)


train_input *= grid_size
train_output *= grid_size
test_input *= grid_size
test_output *= grid_size

# build your neural net
if categorial == True:
    model = Sequential([
        LSTM(64, input_dim=2),
        Activation('relu'),
        Dropout(0.15),
        Dense(64),
        Activation('linear'),
        Dense(9),
        Activation('softmax'),
    ])

    # Another way to define your optimizer
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0001, clipnorm=1.)

    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Training ------------')
    # Another way to train the model
    model.fit(train_input, train_dir_output, epochs=10, batch_size=50)

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(test_input, test_dir_output)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

    # serialize model to JSON
    model_json = model.to_json()
    with open("LSTM models/LSTM_25%_data.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("LSTM models/LSTM_25%_data.h5")
    print("Saved model to disk")
    print(test_dir_output[5])

    print(model.predict(np.reshape(test_input[5], (-1, time_steps, 2))[:,:-6,:], batch_size=1))


else:
    model = Sequential([
        LSTM(64, batch_size=(time_steps, 2)),
        Activation('relu'),
        Dropout(0.25),
        Dense(32),
        Activation('linear'),
        Dense(2)
    ])

    # Another way to define your optimizer
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0001, clipnorm=1.)

    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='mse',
                  metrics=['accuracy', 'mae'])

    print('Training ------------')
    # Another way to train the model
    model.fit(train_input, train_output, epochs=10000, batch_size=100)

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy, mae = model.evaluate(test_input, test_output)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)
    print('test mae: ', mae)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_position_prediction_with_more_time_steps.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_position_prediction_with_more_time_steps.h5")
    print("Saved model to disk")
