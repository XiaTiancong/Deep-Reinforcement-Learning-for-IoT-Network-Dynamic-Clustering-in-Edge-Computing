import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import bisect
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils

class AGV:
    def __init__(self, mode, data_proportion = 1):
        self.seq = []
        self.secs = []
        self.nsecs = []
        self.x = []
        self.y = []
        self.z = []
        self.w = []
        self.record_position = False
        self.record_orientation = False
        self.data_proportion = data_proportion
        if mode == 'training':
            for i in range(1,7):
                self.f = open("../agv_data/agv1_pose" + str(i) + ".log", "r")
                for line in self.f:
                    if "seq:" in line:
                        seq_pattern = re.compile('(?<=seq: )\d*')
                        seq_num = seq_pattern.findall(line)
                        self.seq.append(int(seq_num[0]))
                    if ' secs' in line:
                        secs_pattern = re.compile('(?<= secs: )\d*')
                        secs_val = secs_pattern.findall(line)
                        if secs_val:
                            self.secs.append(int(secs_val[0]))
                    if "position:" in line:
                        self.record_position = True
                    if self.record_position:
                        if "x:" in line:
                            x_pattern = re.compile('(?<=x: )-?\d*\.?\d*')
                            x_pos = x_pattern.findall(line)
                            self.x.append(float(x_pos[0]))
                        if "y:" in line:
                            y_pattern = re.compile('(?<=y: )-?\d*\.?\d*')
                            y_pos = y_pattern.findall(line)
                            self.y.append(float(y_pos[0]))
                            self.record_position = False
                    if "orientation:" in line:
                        self.record_orientation = True
                    if self.record_orientation:
                        if "z:" in line:
                            z_pattern = re.compile('(?<=z: )-?\d*\.?\d*')
                            z_pos = z_pattern.findall(line)
                            self.z.append(float(z_pos[0]))
                        if "w:" in line:
                            w_pattern = re.compile('(?<=w: )-?\d*\.?\d*')
                            w_pos = w_pattern.findall(line)
                            self.w.append(float(w_pos[0]))
                            self.record_orientation = False
                    if 'nsecs' in line:
                        nsecs_pattern = re.compile('(?<= nsecs: )\d*')
                        nsecs_val = nsecs_pattern.findall(line)
                        if nsecs_val:
                            self.nsecs.append(int(nsecs_val[0]))
                self.f.close()
        if mode == 'training_with_insufficient_data':
            for i in range(2,4):
                self.f = open("../agv_data/agv1_pose" + str(i) + ".log", "r")
                for line in self.f:
                    if "seq:" in line:
                        seq_pattern = re.compile('(?<=seq: )\d*')
                        seq_num = seq_pattern.findall(line)
                        self.seq.append(int(seq_num[0]))
                    if ' secs' in line:
                        secs_pattern = re.compile('(?<= secs: )\d*')
                        secs_val = secs_pattern.findall(line)
                        if secs_val:
                            self.secs.append(int(secs_val[0]))
                    if "position:" in line:
                        self.record_position = True
                    if self.record_position:
                        if "x:" in line:
                            x_pattern = re.compile('(?<=x: )-?\d*\.?\d*')
                            x_pos = x_pattern.findall(line)
                            self.x.append(float(x_pos[0]))
                        if "y:" in line:
                            y_pattern = re.compile('(?<=y: )-?\d*\.?\d*')
                            y_pos = y_pattern.findall(line)
                            self.y.append(float(y_pos[0]))
                            self.record_position = False
                    if "orientation:" in line:
                        self.record_orientation = True
                    if self.record_orientation:
                        if "z:" in line:
                            z_pattern = re.compile('(?<=z: )-?\d*\.?\d*')
                            z_pos = z_pattern.findall(line)
                            self.z.append(float(z_pos[0]))
                        if "w:" in line:
                            w_pattern = re.compile('(?<=w: )-?\d*\.?\d*')
                            w_pos = w_pattern.findall(line)
                            self.w.append(float(w_pos[0]))
                            self.record_orientation = False
                    if 'nsecs' in line:
                        nsecs_pattern = re.compile('(?<= nsecs: )\d*')
                        nsecs_val = nsecs_pattern.findall(line)
                        if nsecs_val:
                            self.nsecs.append(int(nsecs_val[0]))

                self.f.close()
        if mode == 'testing':
            for i in range(5,9):
                self.f = open("../agv_data/agv1_pose" + str(i) + ".log", "r")
                for line in self.f:
                    if "seq:" in line:
                        seq_pattern = re.compile('(?<=seq: )\d*')
                        seq_num = seq_pattern.findall(line)
                        self.seq.append(int(seq_num[0]))
                    if ' secs' in line:
                        secs_pattern = re.compile('(?<= secs: )\d*')
                        secs_val = secs_pattern.findall(line)
                        if secs_val:
                            self.secs.append(int(secs_val[0]))
                    if "position:" in line:
                        self.record_position = True
                    if self.record_position:
                        if "x:" in line:
                            x_pattern = re.compile('(?<=x: )-?\d*\.?\d*')
                            x_pos = x_pattern.findall(line)
                            self.x.append(float(x_pos[0]))
                        if "y:" in line:
                            y_pattern = re.compile('(?<=y: )-?\d*\.?\d*')
                            y_pos = y_pattern.findall(line)
                            self.y.append(float(y_pos[0]))
                            self.record_position = False
                    if "orientation:" in line:
                        self.record_orientation = True
                    if self.record_orientation:
                        if "z:" in line:
                            z_pattern = re.compile('(?<=z: )-?\d*\.?\d*')
                            z_pos = z_pattern.findall(line)
                            self.z.append(float(z_pos[0]))
                        if "w:" in line:
                            w_pattern = re.compile('(?<=w: )-?\d*\.?\d*')
                            w_pos = w_pattern.findall(line)
                            self.w.append(float(w_pos[0]))
                            self.record_orientation = False
                    if 'nsecs' in line:
                        nsecs_pattern = re.compile('(?<= nsecs: )\d*')
                        nsecs_val = nsecs_pattern.findall(line)
                        if nsecs_val:
                            self.nsecs.append(int(nsecs_val[0]))
                self.f.close()
        self.secs = np.asarray(self.secs)
        self.nsecs = np.asarray(self.nsecs)
        self.time = self.secs + self.nsecs*(1e-9)
        self.time = self.time - self.time[2]
        self.df = pd.DataFrame({'time': self.time,
                           'pos_x': self.x,


                           'pos_y': self.y,
                           })
        self.df = self.df[2:]
        self.df.index = np.arange(len(self.df))
        df_shift = self.df.shift(1)
        self.diff = self.df - df_shift
        self.df['time_diff'] = self.diff.to_dict(orient='list')['time']
        distance = list(map(lambda x, y: (x ** 2 + y ** 2) ** 1 / 2, self.diff['pos_x'], self.diff['pos_y']))
        self.df['distance'] = distance
        speed_before = np.asarray(distance) / np.asarray(self.df['time_diff'])
        self.df['speed_before'] = speed_before
        self.df['speed_x'] = np.asarray(self.diff.to_dict(orient='list')['pos_x']) / np.asarray(self.df['time_diff'])
        self.df['speed_y'] = np.asarray(self.diff.to_dict(orient='list')['pos_y']) / np.asarray(self.df['time_diff'])

    def get_df(self):
        self.df = self.df[:round(self.df.shape[0]*self.data_proportion)]
        return self.df

    def identical_sample_rate(self, df, sample_period):
        df = df.to_dict(orient='list')
        time = df['time']
        x = df['pos_x']
        y = df['pos_y']
        speed_before_x = df['speed_x']
        speed_before_y = df['speed_y']

        distance = df['distance']
        new_df = pd.DataFrame()
        new_time = [i * sample_period for i in range(0, int((time[-1]-time[0])//sample_period)-1)]
        new_x = []
        new_y = []
        new_speed_x = []
        new_speed_y = []
        for new_sec in new_time:
            previous_recorded = bisect.bisect(time, new_sec) - 1
            new_x_next = ((x[previous_recorded+1] - x[previous_recorded])/(time[previous_recorded+1] - time[previous_recorded]))*(new_sec - time[previous_recorded]) + x[previous_recorded]
            new_x.append(new_x_next)

            new_y_next = ((y[previous_recorded + 1] - y[previous_recorded]) / (
                        time[previous_recorded + 1] - time[previous_recorded])) * (new_sec - time[previous_recorded]) + y[
                             previous_recorded]
            new_y.append(new_y_next)



            new_speed_x_next = speed_before_x[previous_recorded+1]
            new_speed_x.append(new_speed_x_next)

            new_speed_y_next = speed_before_y[previous_recorded + 1]
            new_speed_y.append(new_speed_y_next)

        new_df['time'] = new_time
        new_df['pos_x'] = new_x
        new_df['pos_y'] = new_y
        new_df['speed_x'] = new_speed_x
        new_df['speed_y'] = new_speed_y

        return new_df
    '''
    def get_direction(self, dir_x, dir_y, threshold):
        dir_x = np.asarray(dir_x).reshape(-1,1)
        dir_y = np.asarray(dir_y).reshape(-1,1)
        direction_metrix = np.concatenate((dir_x, dir_y), axis=1)
        dir_output = np.zeros((direction_metrix.shape[0],))

        for i in range(np.shape(direction_metrix)[0]):
            if abs(direction_metrix[i,0]) <= threshold:
                if abs(direction_metrix[i, 1]) <= threshold:
                    dir_output[i] = 4           #stay
                elif direction_metrix[i,1] + threshold < 0:
                    dir_output[i] = 3            # downs
                elif direction_metrix[i,1] - threshold > 0:   # up
                    dir_output[i] = 5
                else:
                    print("category division in the 1st situation is wrong")
            elif abs(direction_metrix[i,1]) <= threshold:
                if direction_metrix[i,0] + threshold < 0:    # left
                    dir_output[i] = 1
                elif direction_metrix[i,0] - threshold > 0: # right
                    dir_output[i] = 7
                else:
                    print("category division in the 2nd situation is wrong")
            else:
                if direction_metrix[i,0] + threshold < 0 and direction_metrix[i,1] + threshold < 0:   # downleft
                    dir_output[i] = 0
                elif direction_metrix[i,0] + threshold < 0 and direction_metrix[i,1] - threshold > 0:  # upleft
                    dir_output[i] = 2
                elif direction_metrix[i,0] - threshold > 0 and direction_metrix[i,1] + threshold < 0:   # downright
                    dir_output[i] = 6
                elif direction_metrix[i,0] - threshold > 0 and direction_metrix[i,1] - threshold > 0:  # upright
                    dir_output[i] = 8
                else:
                    print("category division in the 3rd situation is wrong")
        return dir_output
        

    def add_behavior(self, df, threshold):
        df['dir'] = np.asarray((self.get_direction(df['speed_x'], df['speed_y'], threshold)))
        df['dir'] = df['dir'].shift(-1)     # turn experience replay to prediction of the behavior
        df = df[:-1]
        return df                                   
    print(test_dir_output[5])
    print(model.predict(np.reshape(test_input[5], (-1, time_steps, 2))[:,:-6,:], batch_size=1))    #keep in mind that threshold and speed is not fit to canvas
        '''

    def fit_to_canvas(self, df, scale):

        x = df.to_dict(orient='list')['pos_x']
        extreme_points = [[max(x)], [min(x)]]
        scaler = MinMaxScaler(feature_range=(0, scale))
        scaler.fit(extreme_points)
        df['pos_x'] = scaler.transform(np.reshape(x, (-1, 1)))

        y = df.to_dict(orient='list')['pos_y']
        extreme_points_y = [[max(y)], [min(y)]]
        scaler_y = MinMaxScaler(feature_range=(0, scale))
        scaler_y.fit(extreme_points_y)
        df['pos_y'] = scaler_y.transform(np.reshape(y, (-1,1)))
        return df

    def trace_grided(self, df, grid_size):
        a = df.to_dict(orient = 'list')['pos_x']
        b = df.to_dict(orient = 'list')['pos_y']
        a = list(map(lambda x: int(round(x/grid_size)), a))
        b = list(map(lambda x: int(round(x / grid_size)), b))
        a = np.asarray(a).reshape(-1,1)
        b = np.asarray(b).reshape(-1,1)
        result = np.concatenate((a,b),axis=1)
        return result

    def delete_staying_pos(self, pos_grided):
        trace_grided = [pos_grided[0]]
        last_item = pos_grided[0]
        for item in pos_grided[1:]:
            if (item == last_item).all():
                continue
            else:
                x_distance = int((item - last_item)[0])
                y_distance = int((item - last_item)[1])
                diagonal_distance = min(abs(x_distance), abs(y_distance))
                for i in range(int(diagonal_distance)):
                    last_item = last_item + np.asarray([x_distance/abs(x_distance), y_distance/abs(y_distance)])
                    trace_grided.append(last_item)
                if abs(x_distance) > diagonal_distance:
                    for j in range(abs(x_distance) - diagonal_distance):
                        last_item = last_item + np.asarray([x_distance/abs(x_distance), 0])
                        trace_grided.append(last_item)
                elif abs(y_distance) > diagonal_distance:
                    for j in range(abs(y_distance) - diagonal_distance):
                        last_item = last_item + np.asarray([0, y_distance/abs(y_distance)])
                        trace_grided.append(last_item)
        return trace_grided

    def get_inputs_and_outputs(self, pos_list, input_num):
        total_points_num = len(pos_list)
        inputs = []
        outputs = []
        for i in range(total_points_num - input_num):
            inputs.append(pos_list[i:i + input_num])
            outputs.append(pos_list[i + input_num])
        return np.asarray(inputs), np.asarray(outputs)

    def get_direction(self, direction_metrix):
        train_dir_output = np.zeros((direction_metrix.shape[0],))
        for i in range(np.shape(direction_metrix)[0]):
            if list(direction_metrix[i]) == [-1, -1]:  # downleft
                train_dir_output[i] = 0
            elif list(direction_metrix[i]) == [-1, 0]:  # left
                train_dir_output[i] = 1
            elif list(direction_metrix[i]) == [-1, 1]:  # upleft
                train_dir_output[i] = 2
            elif list(direction_metrix[i]) == [0, -1]:  # down
                train_dir_output[i] = 3
            elif list(direction_metrix[i]) == [0, 0]:  # stay
                train_dir_output[i] = 4
            elif list(direction_metrix[i]) == [0, 1]:  # up
                train_dir_output[i] = 5
            elif list(direction_metrix[i]) == [1, -1]:  # downright
                train_dir_output[i] = 6
            elif list(direction_metrix[i]) == [1, 0]:  # right
                train_dir_output[i] = 7
            elif list(direction_metrix[i]) == [1, 1]:  # upright
                train_dir_output[i] = 8
            else:
                return
        return train_dir_output

    def compare(self, x, y):
        if x > y:
            return 1
        elif x < y:
            return -1
        else:
            return 0

    def get_dir(self, input_data, output_data):
        direction_metrix = map(self.compare, np.reshape(output_data, (-1,)), np.reshape(input_data[:, -1], (-1,)))
        direction_metrix = np.asarray(list(direction_metrix)).reshape(-1, 2)
        dir_output = self.get_direction(direction_metrix)
        dir_output = np_utils.to_categorical(dir_output, num_classes=9)
        return dir_output

if __name__ == "__main__":
    threshold = 0.3
    sample_period = 0.1
    grid_size = 0.025
    agv = AGV("training_with_insufficient_data")
    df = agv.get_df()
    df = agv.fit_to_canvas(df, 15)
    df_sampled = agv.identical_sample_rate(df, sample_period)
    grided_pos = agv.trace_grided(df_sampled, grid_size)
    trace_grided = agv.delete_staying_pos(grided_pos)

    time_steps = 60
    train_propotion = 0.8
    train_set_num = int(round(len(trace_grided) * train_propotion))
    train_set = trace_grided[:]  # train_set_num]
    test_set = trace_grided[train_set_num:]

    train_input, train_output = agv.get_inputs_and_outputs(train_set, time_steps)
    test_input, test_output = agv.get_inputs_and_outputs(test_set, time_steps)
    train_dir_output = agv.get_dir(train_input, train_output)
    test_dir_output = agv.get_dir(test_input, test_output)
    print('Finish importing all the data :)')
    print(len(train_dir_output))
    '''
    plt.plot(df_sampled['time'], df_sampled['dir'])
    plt.xlabel('secs')
    plt.ylabel('dir')
    plt.show()
    '''

    plt.scatter(df['time'], df['pos_x'])
    plt.plot(df['time'], df['pos_y'])
    #plt.plot(df_sampled['time'], df_sampled['dir'])
    #plt.plot(df.index, df['ori_z'])
    #plt.plot(df.index, df['ori_w'])
    plt.xlabel('secs')
    plt.ylabel('x,y_pos')
    plt.show()
