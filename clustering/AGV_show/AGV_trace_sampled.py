import turtle
import random
from read_AGV_data import AGV
import numpy as np
import time

s = turtle.Screen()
ball = turtle.Turtle()
s.title('Ball Bouncing in a Box')
s.bgcolor("black")
s.tracer(1)
ball.color("white")
ball.speed(-1)
ball.shape('circle')
ball.hideturtle()



############ Defining Functions ############

# Function to draw borders
def drawRect():
    ball.pensize(10)
    ball.pendown()
    ball.goto(500, 0)
    ball.goto(500, 300)
    ball.goto(0, 300)
    ball.goto(0, 0)
    ball.penup()


# Main Movement Function
def mainMove():
    sample_period = 0.3
    agv = AGV()
    df = agv.get_df()
    df = agv.identical_sample_rate(df, sample_period)
    data = df.to_dict(orient='list')
    sec = data['time']
    x = 10 * np.asarray(data['pos_x'])+ 150
    y = 10 * np.asarray(data['pos_y']) + 150
    predict_mode = False
    ball.goto(x[0], y[0])
    ball.showturtle()
    ball.pensize(2)
    ball.pendown()
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
    for i in range(1, len(sec)):
        # Just for smoothness' sake
        s.update()
        # Calls function to detect pockets (Optional)
        # pocketDetect1()
        # pocketDetect2()
        # pocketDetect3()
        # pocketDetect4()
        # Actually moves the ball

        if not predict_mode:
            ball.speed(0)
            ball.setx(x[i])
            ball.sety(y[i])
        else:
            pass

        # Checks for Keyboard Input
        #turtle.listen()
        #turtle.onkey(exit, "x")
        turtle.onkey(reset, "r")
        #turtle.onkey(red, "1")
        #turtle.onkey(orange, "2")
        #urtle.onkey(yellow, "3")
        #turtle.onkey(green, "4")
        #turtle.onkey(blue, "5")
        #turtle.onkey(purple, "6")
        #turtle.onkey(white, "7")
        #turtle.onkey(black, "8")
        # Checks for Collision



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