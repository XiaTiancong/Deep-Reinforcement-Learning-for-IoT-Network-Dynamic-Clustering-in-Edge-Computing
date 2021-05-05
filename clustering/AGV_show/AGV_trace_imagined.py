import turtle
import random
from read_AGV_data import AGV
import numpy as np
from keras.models import model_from_json
import time

s = turtle.Screen()
ball = turtle.Turtle()
s.title('Ball Bouncing in a Box')
s.bgcolor("white")
s.tracer(1)
ball.color("black")
ball.speed(-1)
ball.shape('circle')
ball.hideturtle()
RNN_json_file = open('../LSTM models/LSTM_full_data.json', 'r')
loaded_model_json_ = RNN_json_file.read()
RNN_json_file.close()
RNN_model = model_from_json(loaded_model_json_)
# load weights into new model
RNN_model.load_weights("../LSTM models/LSTM_full_data.h5")
print("Loaded RNN model from disk")
time_steps = 60
pos_buf = []
############ Defining Functions ############

def get_position_update(behavior):
    if behavior == 0:
        position_update = [-1, -1]
    elif behavior == 1:
        position_update = [-1, 0]
    elif behavior == 2:
        position_update = [-1, 1]
    elif behavior == 3:
        position_update = [0, -1]
    elif behavior == 4:
        position_update = [0, 0]
    elif behavior == 5:
        position_update = [0, 1]
    elif behavior == 6:
        position_update = [1, -1]
    elif behavior == 7:
        position_update = [1, 0]
    elif behavior == 8:
        position_update = [1, 1]

    return position_update

# Function to draw borders
def drawRect():
    ball.pensize(5)
    ball.pendown()
    ball.goto(300, 0)
    ball.goto(300, 300)
    ball.goto(0, 300)
    ball.goto(0, 0)
    ball.penup()
    ball.color("blue")


# Main Movement Function
def mainMove():
    sample_period = 0.3
    grid_size = 0.025
    agv = AGV('training')
    df = agv.get_df()
    df = agv.fit_to_canvas(df, 15)
    df_sampled = agv.identical_sample_rate(df, sample_period)
    grided_pos = agv.trace_grided(df_sampled, grid_size).astype(float)
    trace_grided = agv.delete_staying_pos(grided_pos)
    trace_grided = np.asarray(trace_grided)
    trace_grided *= grid_size

    x = trace_grided[:, 0]
    y = trace_grided[:, 1]
    starting_point = np.random.randint(len(x))
    ball.penup()
    ball.goto(x[starting_point]*20, y[starting_point]*20)
    ball.showturtle()
    ball.pensize(2)
    ball.pendown()
    random_po_counter = 0
    behavior_prediction = np.random.randint(9)
    '''
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    '''


    total_counter = 0
    bounce_counter = 0
    for j in range(1, len(x)*15):
        # Just for smoothness' sake
        s.update()
        # Calls function to detect pockets (Optional)
        # pocketDetect1()
        # pocketDetect2()
        # pocketDetect3()
        # pocketDetect4()
        # Actually moves the ball
        i = ((starting_point) + j) % len(x)
        total_counter += 1
        if j <= 200:
            ball.speed(10)
            x_pos = x[i]
            y_pos = y[i]
        else:
            behavior_prediction_pos = RNN_model.predict(
                np.reshape(pos_buf, (1, -1, 2)), batch_size=1)
            randCol = ('Orange')
            ball.color(randCol)
            pos_sum = sum(behavior_prediction_pos[0])
            choose_action = pos_sum * np.random.random()
            if np.random.random() < 1:
                if choose_action <= sum(behavior_prediction_pos[0,:1]):
                    behavior_prediction = 0
                elif choose_action <= sum(behavior_prediction_pos[0,:2]):
                    behavior_prediction = 1
                elif choose_action <= sum(behavior_prediction_pos[0,:3]):
                    behavior_prediction = 2
                elif choose_action <= sum(behavior_prediction_pos[0,:4]):
                    behavior_prediction = 3
                elif choose_action <= sum(behavior_prediction_pos[0,:5]):
                    behavior_prediction = 4
                elif choose_action <= sum(behavior_prediction_pos[0,:6]):
                    behavior_prediction = 5
                elif choose_action <= sum(behavior_prediction_pos[0,:7]):
                    behavior_prediction = 6
                elif choose_action <= sum(behavior_prediction_pos[0,:8]):
                    behavior_prediction = 7
                elif choose_action <= sum(behavior_prediction_pos[0,:9]):
                    behavior_prediction = 8
            else:
                random_po_counter += 1
                if random_po_counter == 2:
                    behavior_prediction = np.random.randint(9)
                    random_po_counter = 0
            position_update = get_position_update(behavior_prediction)
            x_pos += position_update[0] * grid_size
            y_pos += position_update[1] * grid_size
            if x_pos >= 15:
                x_pos -= 2 * grid_size
                bounce_counter += 1
            if x_pos <= 0:
                x_pos += 2 * grid_size
                bounce_counter += 1
            if y_pos >= 15:
                y_pos -= 2 * grid_size
                bounce_counter += 1
            if y_pos <= 0:
                y_pos += 2 * grid_size
                bounce_counter += 1
            if bounce_counter >= 15:
                print("stuck")
            if total_counter % 40 == 0:
                bounce_counter = 0
                print(total_counter)

        if len(pos_buf) < 60:
            pos_buf.append([x_pos, y_pos])
        else:
            pos_buf.pop(0)
            pos_buf.append([x_pos, y_pos])
        ball.setx(x_pos*20)
        ball.sety(y_pos*20)


# Functions to Reset and Exit the program
def exit():
    s.bye()


def reset():
    global position,position_queue
    ball.penup()
    ball.speed(0)
    ball.setpos(250, 150)
    ball.pendown()
    ball.showturtle()
    collider()
    position = [ball.xcor(), ball.ycor()]
    position_queue = []
    mainMove()


# Detect if the ball is within a 'Pocket' (This will 'delete' the ball)
def pocketDetect1():
    if 0 < ball.ycor() < 30 and 0 < ball.xcor() < 30:
        ball.hideturtle()
        ball.penup()
        ball.dy = 0
        ball.dx = 0


def pocketDetect2():
    if 0 < ball.ycor() < 30 and 470 < ball.xcor() < 500:
        ball.hideturtle()
        ball.penup()
        ball.dy = 0
        ball.dx = 0


def pocketDetect3():
    if 270 < ball.ycor() < 300 and 0 < ball.xcor() < 30:
        ball.hideturtle()
        ball.penup()
        ball.dy = 0
        ball.dx = 0


def pocketDetect4():
    if 270 < ball.ycor() < 300 and 470 < ball.xcor() < 500:
        ball.hideturtle()
        ball.penup()
        ball.dy = 0
        ball.dx = 0


# This makes it so the ball changes color every time it collides with the wall
def collider():
    randCol = ('')
    colorInt = random.randint(0, 5)
    if colorInt == 0:
        randCol = ('Red')
    elif colorInt == 1:
        randCol = ('Orange')
    elif colorInt == 2:
        randCol = ('Yellow')
    elif colorInt == 3:
        randCol = ('Green')
    elif colorInt == 4:
        randCol = ('Blue')
    elif colorInt == 5:
        randCol = ('Purple')
    ball.color(randCol)


# Functions for Changing Background Color
def red():
    s.bgcolor('Red')


def orange():
    s.bgcolor('Orange')


def yellow():
    s.bgcolor('Yellow')


def green():
    s.bgcolor('Green')


def blue():
    s.bgcolor('Blue')


def purple():
    s.bgcolor('Purple')


def white():
    s.bgcolor('White')


def black():
    s.bgcolor('Black')


# Running the functions
drawRect()
mainMove()