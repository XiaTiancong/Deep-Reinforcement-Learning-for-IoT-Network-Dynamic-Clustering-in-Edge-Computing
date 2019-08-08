import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import keras
import numpy as np
import random
import json
import networkx as nx
import matplotlib.pyplot as plt
import operator
import math
from functools import reduce
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import sgd
from keras.models import model_from_json
from keras.optimizers import Adam
from read_AGV_data import AGV

class Cluster(object):
    def __init__(self):
        self.init_states()
        self.setup_network()
        self.setup_parameters()

    def init_states(self):
        self.state_G=nx.Graph()
        self.state_xcor = []
        self.state_ycor = []
        self.state_link = []
        self.state_server = []
        self.state_cluster_id = []
        self.state_cluster_core_xcor = []
        self.state_cluster_core_ycor = []
        self.state_event_xcor = []
        self.state_event_ycor = []
        self.state_event_step_rate_xcor = 0
        self.state_event_step_rate_ycor = 0

    def reuse_network(self, s_xcor, s_ycor):
        self.reuse_network_topology(s_xcor, s_ycor)

    def setup_network(self):
        self.set_network_topology()
        while self.all_nodes_connected() == False:
            self.set_network_topology()
    
    def setup_parameters(self):        
        self.set_server_and_header()
        self.set_cluster_id()
        self.set_events()
    
    def draw_network(self):
        pos=nx.kamada_kawai_layout(self.state_G)
        nx.draw(self.state_G, pos, with_labels=True, cmap=plt.get_cmap('Accent'), node_color=self.state_cluster_id, node_size=200)
        plt.show()        
    
	# Make sure the distance between neighbor nodes is larger than a required value. 
    def check_neighbor_distance_larger_than_min_range(self, node_id):
        good_position = 1
        for j in range(0, node_id):
            ax = self.state_xcor[node_id]
            ay = self.state_ycor[node_id]
            bx = self.state_xcor[j]
            by = self.state_ycor[j]
            distance_square = (ax-bx)**2 + (ay-by)**2
            if distance_square < min_distance_between_nodes_square:
                good_position = 0
        
        return good_position
    
	# Deploy the nodes in random positions. 
    def scatter_node_random_position(self):
        for i in range(0, total_node_number):
            good_position = 0
            for find_good_position_time in range(0, max_find_good_position_time):
                if good_position == 0:
                    self.state_xcor[i] = random.random() * deploy_range_x
                    self.state_ycor[i] = random.random() * deploy_range_y
                    good_position = self.check_neighbor_distance_larger_than_min_range(i)
    
	# The state_link is the connectivity matrix of the network. 
    def set_network_connectivity(self):
        self.state_link = []
        transmit_range_square = transmit_range ** 2
        for i in range(0, node_number+server_number):
            node_link = []
            for j in range(0, node_number+server_number):
                if i!=j and (self.state_xcor[i]-self.state_xcor[j])**2 + (self.state_ycor[i]-self.state_ycor[j])**2 <= transmit_range_square:
                    node_link.append(1)
                else:
                    node_link.append(0)
            self.state_link.append(node_link)
        self.set_graph()
    
    def reuse_network_topology(self, s_xcor, s_ycor):
        self.state_xcor = s_xcor
        self.state_ycor = s_ycor
        self.set_network_connectivity()
    
	# Use the saved coordinate values or not. 
    def set_network_topology(self):
        '''
        self.state_xcor = []
        self.state_ycor = [] 
        for i in range(0, node_number+server_number):
            self.state_xcor.append(0)
            self.state_ycor.append(0)
        '''
        self.state_xcor = np.zeros((node_number+server_number,))
        self.state_ycor = np.zeros((node_number+server_number,))

        if flag_benchmark_topology == 1:
            self.scatter_node_random_position()
            with open('state_xcor.txt', 'w') as f:
                for item in self.state_xcor:
                    f.write("%s\n" % item)
            f.close()
            with open('state_ycor.txt', 'w') as f:
                for item in self.state_ycor:
                    f.write("%s\n" % item)
            f.close()
        else:
            f = open("state_xcor.txt", "r")
            i = 0
            for x in f:
                self.state_xcor[i] = float(x.strip())
                i = i + 1
            f.close()
            f = open("state_ycor.txt", "r")
            i = 0
            for y in f:
                self.state_ycor[i] = float(y.strip())
                i = i + 1
            f.close()
        
        self.set_network_connectivity()
    
	# The positions and ID of cluster headers are initialized. 
    def set_server_and_header(self):
        self.state_server = []
        for i in range(0, server_number):
            self.state_server.append(i)
            self.state_cluster_core_xcor.append(0)
            self.state_cluster_core_ycor.append(0)
            self.state_cluster_core_xcor[i] = self.state_xcor[i]
            self.state_cluster_core_ycor[i] = self.state_ycor[i]
    
	# Select the node that is closest to the moving cluster-core as the cluster header. Use this cluster header to make voronoi clusters. 
    def set_cluster_id(self):
        self.state_cluster_id = [0] * (node_number+server_number)
        temp_cluster_proxy = []
        
        for i in range(0, server_number):
            self.state_cluster_id[i] = self.state_server[i]
            temp_cluster_proxy.append(0)
            min_distance_square = (deploy_range_x + deploy_range_y)**2
            min_id = -1
            for j in range(server_number, node_number+server_number):
                temp_distance_square = (self.state_cluster_core_xcor[i]-self.state_xcor[j])**2 + (self.state_cluster_core_ycor[i]-self.state_ycor[j])**2
                if temp_distance_square < min_distance_square:
                    min_distance_square = temp_distance_square
                    min_id = j
            temp_cluster_proxy[i] = min_id

        for i in range(server_number, node_number+server_number):
            closest_header_cluster_id = -1
            closest_distance = node_number
            for j in range(0, server_number):
                header_id = temp_cluster_proxy[j]
                hop_distance = len(self.find_route(i, header_id)) - 1
                if hop_distance < closest_distance:
                    closest_header_cluster_id = self.state_cluster_id[j]
                    closest_distance = hop_distance                                #?? here the hop_distance is not used after this
            self.state_cluster_id[i] = closest_header_cluster_id
    
	# In the network init stage, make sure all nodes are connected to the network. 
    def all_nodes_connected(self):
        for i in range(0, node_number+server_number):
            for j in range(0, node_number+server_number):
                check = nx.has_path(self.state_G, i, j)
                if check == False:
                    return False
        return True    
    
	# Init graph parameters for python graph library. 
    def set_graph(self):
        self.state_G=nx.Graph()
        for i in range(0, node_number+server_number):
            self.state_G.add_node(i)
        for i in range(0, node_number+server_number):
            for j in range(i, node_number+server_number):
                if self.state_link[i][j] == 1 and self.state_link[j][i] == 1:
                    self.state_G.add_edge(i, j)
    
	# Find the path from source node "s" to the destination node "t". 
    def find_route(self, s, t):
        check = nx.has_path(self.state_G, source=s, target=t)
        if check == True:
            path = nx.dijkstra_path(self.state_G, source=s, target=t)
        else:
            path = []        
        return path
    
	# Every time tick, we run this function. This function generates data from each node to the cluster servers.
    # If the senser is in the detection range of moving event, the senser generates more data.
    def transmit_flow_in_network(self):
        state_cluster_path_length = [0] * server_number
        state_cluster_size = [0] * server_number
        for i in range(server_number, total_node_number):
            j = self.state_cluster_id[i]
            pass_route = self.find_route(i, j)
            event_distance_square = (self.state_xcor[i] - self.state_event_xcor)**2 + (self.state_ycor[i] - self.state_event_ycor)**2
                    
            if event_distance_square < event_detection_range_square:
                state_cluster_size[j] = state_cluster_size[j] + 1 * event_data_increase_rate
                state_cluster_path_length[j] = state_cluster_path_length[j] + len(pass_route) * event_data_increase_rate
            else:
                state_cluster_size[j] = state_cluster_size[j] + 1
                state_cluster_path_length[j] = state_cluster_path_length[j] + len(pass_route)
        
		# To calculate the reward values, please refer the paper. 
        mean_state_balance_data = np.mean(state_cluster_size)
        total_state_balance = 0
        for k in range(0, len(state_cluster_size)):
            total_state_balance = total_state_balance + (1 - abs(state_cluster_size[k] - mean_state_balance_data) / mean_state_balance_data)
        state_balance_mse_data = total_state_balance / len(state_cluster_size)
        
		# To calculate the reward values, please refer the paper. 
        mean_state_balance_com = np.mean(state_cluster_path_length)
        total_state_balance = 0
        for k in range(0, len(state_cluster_path_length)):
            total_state_balance = total_state_balance + (1 - abs(state_cluster_path_length[k] - mean_state_balance_com) / mean_state_balance_com)
        state_balance_mse_com = total_state_balance / len(state_cluster_path_length)
        
        return (state_balance_mse_data*state_balance_mse_com)
    
	# This function generate a vector, each value of the vector represents how much data size the senser is producing at this time tick. 
    def workload_in_network(self):
        state_workload = [0] * total_node_number
        for i in range(server_number, total_node_number):
            event_distance_square = (self.state_xcor[i] - self.state_event_xcor)**2 + (self.state_ycor[i] - self.state_event_ycor)**2
            if event_distance_square < event_detection_range_square:
                state_workload[i] = 1 * event_data_increase_rate
            else:
                state_workload[i] = 1
        return state_workload
    
	# If in the gameover mode, we check how much the clusters are unbalanced. If the unbalanced value is over a threshold, then game over. 
    def check_flow_in_network_fail(self):
        balance = self.transmit_flow_in_network()
        
        if balance < max_balance_diff:
            return 1
        else:
            return 0
    
	# Make sure the graph matrix is correctly created. The network is bi-direction, so the matrix should be symmetrical。 
    def check_graph_error(self):
        for i in range(0, node_number+server_number):
            for j in range(i, node_number+server_number):
                if self.state_link[i][j] != self.state_link[j][i]:
                    print("Error: node network topology.")
                    exit()
    
	# For each action number, move the position of cluster-core. 
    def action_route(self, action, tick):
        # [5 actions]: stay, up, down, left, right.
        cluster_id = math.floor(action / model_action_number)
        action_id = action % model_action_number
        
        if action_id == 1:
            if self.state_cluster_core_ycor[cluster_id] + move_step_length_header <= deploy_range_y:
                self.state_cluster_core_ycor[cluster_id] = self.state_cluster_core_ycor[cluster_id] + move_step_length_header
        elif action_id == 2:
            if self.state_cluster_core_ycor[cluster_id] - move_step_length_header >= 0:
                self.state_cluster_core_ycor[cluster_id] = self.state_cluster_core_ycor[cluster_id] - move_step_length_header
        elif action_id == 3:
            if self.state_cluster_core_xcor[cluster_id] + move_step_length_header <= deploy_range_x:
                self.state_cluster_core_xcor[cluster_id] = self.state_cluster_core_xcor[cluster_id] + move_step_length_header
        elif action_id == 4:
            if self.state_cluster_core_xcor[cluster_id] - move_step_length_header >= 0:
                self.state_cluster_core_xcor[cluster_id] = self.state_cluster_core_xcor[cluster_id] - move_step_length_header
        
        self.set_cluster_id()
        self.set_graph()
        self.check_graph_error()
    
    def get_reward(self):
        reward = self.transmit_flow_in_network()
        return reward
    
	# The same as normal DQN procedures. 
    def choose_action(self, p_state):
        if np.random.rand() <= exp_replay.epsilon or len(p_state) == 0:
            action = np.random.randint(0, model_output_size)
        else:
            q = exp_replay.model.predict(p_state)
            action = np.argmax(q[0])
        
        return action
    
	# The state in the DQN includes: state_link, cluster_id, and workload. 
    def get_state(self, tick):
        state = []
        for i in range(0, len(self.state_link)):
            state = state + self.state_link[i]
        state = state + self.state_cluster_id
        state = state + self.workload_in_network()
        state = state + self.event_position
        
        self.check_graph_error()
        p_state = np.asarray(state)
        p_state = p_state[np.newaxis]
        return p_state
    
    def act_action(self, p_state, tick):
        if flag_benchmark_action == 1:
            action = 0
        else:
            action = self.choose_action(p_state)
        self.action_route(action, tick)
        
        return action
    
    def act_reward(self, tick):
        reward = self.get_reward()
        p_next_state = self.get_state(tick)
        
        return reward, p_next_state
    
	# Init the event. Theoretically, the event could be init at any places. In the experiment, to make sure that the clusters are somehow balanced, we select a place that is in the middle position of all clusters. 
    def set_events(self):
        if read_AGV:
            self.agv = AGV('testing')
            self.df = self.agv.get_df()
            self.df = self.agv.fit_to_canvas(self.df, deploy_range)
            self.df_sampled = self.agv.identical_sample_rate(self.df, sample_period)
            self.grided_pos = self.agv.trace_grided(self.df_sampled, grid_size)
            self.trace_grided = self.agv.delete_staying_pos(self.grided_pos)
            self.trace_grided = np.asarray(self.trace_grided)

            self.trace_input, self.trace_output = self.agv.get_inputs_and_outputs(self.trace_grided, time_steps)
            self.trace_dir_output = self.agv.get_dir(self.trace_input, self.trace_output)

            self.trace_input *= grid_size
            self.trace_output *= grid_size
            self.trace_grided = self.trace_input[:, :, -1]

            self.act_move_events

        else:
            min_abs_node_id = 0
            min_abs_cluster_nodes = total_node_number**2
            for i in range(0, total_node_number):
                px = self.state_xcor[i]
                py = self.state_ycor[i]
                cluster_nodes_num_in_range = [0] * server_number
                for j in range(0, total_node_number):
                    if (self.state_xcor[j]-px)**2+(self.state_ycor[j]-py)**2 < event_detection_range_square:
                        cluster_id = self.state_cluster_id[j]
                        cluster_nodes_num_in_range[cluster_id] = cluster_nodes_num_in_range[cluster_id] + 1

                mean_cluster_node_num = np.mean(cluster_nodes_num_in_range)
                total_diff_cluster_node_num = 0
                for k in range(0, len(cluster_nodes_num_in_range)):
                    total_diff_cluster_node_num = total_diff_cluster_node_num + abs(cluster_nodes_num_in_range[k] - mean_cluster_node_num)
                if total_diff_cluster_node_num/len(cluster_nodes_num_in_range) < min_abs_cluster_nodes:
                    min_abs_cluster_nodes = total_diff_cluster_node_num/len(cluster_nodes_num_in_range)
                    min_abs_node_id = i

            self.state_event_xcor = self.state_xcor[min_abs_node_id]
            self.state_event_ycor = self.state_ycor[min_abs_node_id]
            self.event_position = [self.state_event_xcor, self.state_event_ycor]

            # Select a direction to move the event.
            if min_abs_node_id < server_number:
                direction_node_id = (min_abs_node_id + 1) % server_number              #????
            else:
                direction_node_id = 1                                                  #????

            # Calculate the step distance of the event based on the event speed.
            w = (self.state_xcor[direction_node_id]-self.state_event_xcor)
            h = (self.state_ycor[direction_node_id]-self.state_event_ycor)
            s = (h**2+w**2)**0.5
            self.state_event_step_rate_xcor = w / s
            self.state_event_step_rate_ycor = h / s
    
	# The event moves in the network. The event triggers the neighbor sensors to produce more data. 
    def act_move_events(self):
        if read_AGV:
            self.state_event_xcor = self.trace_input[total_tick, -1, 0]
            self.state_event_ycor = self.trace_input[total_tick, -1, 1]
            self.behavior = self.trace_dir_output[total_tick]
            self.event_position = [self.state_event_xcor, self.state_event_ycor]

        else:
            self.state_event_xcor = self.state_event_xcor + self.state_event_step_rate_xcor * move_step_length_event
            self.state_event_ycor = self.state_event_ycor + self.state_event_step_rate_ycor * move_step_length_event
            self.event_position = [self.state_event_xcor, self.state_event_ycor]

            if self.state_event_xcor<0 or self.state_event_xcor>deploy_range_x or self.state_event_ycor<0 or self.state_event_ycor>deploy_range_y:
                self.state_event_step_rate_xcor = self.state_event_step_rate_xcor * -1
                self.state_event_step_rate_ycor = self.state_event_step_rate_ycor * -1


class ExperienceReplay(object):
    def __init__(self, epsilon):
        self.memory = list()
        self.model = self._build_model()
        self.epsilon = epsilon

    def _build_model(self):
        model = Sequential()
        model.add(Dense(hidden_size0, input_shape=(model_input_size,), activation='relu'))
        model.add(Dense(hidden_size1, activation='relu'))
        model.add(Dense(hidden_size2, activation='relu'))
        model.add(Dense(model_output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model
    
    def remember(self, state):
        self.memory.append([state])
        if len(self.memory) > max_memory:
            del self.memory[0]
    
    def replay(self):
        minibatch = random.sample(self.memory, batch_size)
        for mem_state in minibatch:
            state, action, reward, next_state, tick, game_over = mem_state[0]
            
            if game_over == 1 or tick == game_time-1:
                target = reward
            else:
                target = (reward + discount * np.amax(self.model.predict(next_state)[0]))
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > epsilon_min:
            self.epsilon = self.epsilon * epsilon_decay

if __name__ == "__main__":
    flag_benchmark_topology = 0		# 0: use coordinates of on the saved files; 1: make new coordinates.
    flag_benchmark_action = 0		# 0: DQN solution; 1: action is always 0. 
    flag_gameover_mode = 0			# 0: the game last for a specific time length; 1: the game ends once the cluster is unbalanced over a threshold. 

    read_AGV = True
    decision_making = True
    threshold = 0.3  # speed threshold for targets
    sample_period = 0.5  # target movement sample rate
    total_tick = 0
    time_steps = 60

	#The setup of IoT network.
    total_node_number = 80
    server_number = 2
    node_number = total_node_number-server_number
    deploy_range = 15
    deploy_range_x = deploy_range
    deploy_range_y = deploy_range
    transmit_range = 3
    min_distance_between_nodes = 2.6
    min_distance_between_nodes_square = min_distance_between_nodes**2
    
    move_step_length_event = 1		# The movement speed of event. 
    move_step_length_header = 1		# The movement speed of cluster header. 
    
    event_detection_range = transmit_range * 2		# The event can be found in the neighbor area of the sensor node. 
    event_detection_range_square = event_detection_range ** 2
    event_data_increase_rate = 3					# The data size once the moving event is detected by the sensors.
    max_find_good_position_time = 5
    max_balance_diff = 0.8							# The game is over once the cluster is unbalanced over this threshold value.
    
	# The setup parameters of DQN. 
    epoch = 1200
    game_time = 50
    learning_rate = 0.0001
    discount = 0.5
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.999
    
	# Actions of the the DQN model. 
    model_action_number = 5
    total_location_number = 2
    model_input_size = total_node_number**2 + total_node_number + total_node_number + total_location_number     # [neighbor connectivity matrix] + [cluster ID of nodes].
    model_output_size = server_number * model_action_number

	# DNN parameters. 
    hidden_size0 = 600
    hidden_size1 = 300  
    hidden_size2 = 150
    max_memory = 2000 
    batch_size = 20

    grid_size = 0.025
    grid_size_x = grid_size
    grid_size_y = grid_size
    
    env = Cluster()		# Init the simulation environment.
    exp_replay = ExperienceReplay(epsilon)
    save_state_xcor = env.state_xcor		# Save the coordinate values, the values are used for benchmark solutions. 
    save_state_ycor = env.state_ycor

    env.set_events()
    for a in range(epoch):
        env.init_states()
        env.reuse_network(save_state_xcor, save_state_ycor)
        env.set_server_and_header()
        env.set_cluster_id()

        mem_state = []
        game_over = 0
        sta_max_reward = 0.0
        sta_sum_reward = 0.0
        sta_ticks = 0

        decision_making = True
        
        for tick in range(0, game_time):				# In every epoch, we move the event for game_time.
            env.act_move_events()
            if decision_making:
                action = np.random.randint(model_output_size)
                env.action_route(action, tick)
                reward, state = env.act_reward(tick)
                mem_state = state
            else:
                action = 0
                env.action_route(action, tick)
                reward, state = env.act_reward(tick)
                mem_state = state
            sta_ticks = sta_ticks + 1
            total_tick += 1
			# Sta values for print. 
            if reward > sta_max_reward:
                sta_max_reward = reward
            if reward < 0.3:
                decision_making = False
            sta_sum_reward = sta_sum_reward + reward 
            print("==>> Epoch {:03d}/{} | Tick {}/{} | Reward {}/{}".format(a, epoch, tick, game_time, round(reward, 2), round(sta_max_reward, 2)))

            if flag_gameover_mode == 1:
                if game_over == 1:
                    break

            env.act_move_events()
        print("Epoch {:03d}/{} | Average Reward {}".format(a, epoch, round(sta_sum_reward/sta_ticks, 2)))

        f = open('DRL_test/dqn_results_static_.txt', 'a+')
        f.write("%s\n" % (sta_sum_reward/sta_ticks))
        f.close()

        print("====================================================================================================================")
