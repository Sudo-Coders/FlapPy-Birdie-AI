from itertools import cycle
import random
import sys
import numpy as np

import pygame
from pygame.locals import *
import json


i=0;
FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 130 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
eta = 0.7
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}
X = np.array([[100]])
# list of all possible players (tuple of 3 positions of flap)
max_score=0
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


try:
    xrange
except NameError:
    xrange = range



def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='sigmoid'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        # for i in range(1, len(layers) - 1):
        #     r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
        #     self.weights.append(r)
        # # output layer - random((2+1, 1)) : 3 x 1
        # r = 2*np.random.random((layers[i] + 1, layers[i+1])) - 1
        with open('weights.json', 'r') as datafile:
            data_load = json.load(datafile)
        theta1 = np.array(data_load['theta1'])
        theta2 = np.array(data_load['theta2'])
        self.weights.append(theta1)
        self.weights.append(theta2)

    def fit(self, X, y, learning_rate=0.4, epochs=3):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 4 == 0: print ('epochs:', k)

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)))      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


# # Renforcement Learning part with neural network

# class Network(object):
#     """docstring for Network"""
#     def __init__(self, sizes):
#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.random.randn(y,1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x)
#                         for x, y in zip(sizes[:-1], sizes[1:])]
                     

#     def feedforward(self,a):
#         for b,w in zip(self.biases,self.weights):
#             a = sigmoid(np.dot(a,w)+b)
#         return a
                


#     def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
#         """Train the neural network using mini-batch stochastic
#         gradient descent.  The ``training_data`` is a list of tuples
#         ``(x, y)`` representing the training inputs and the desired
#         outputs.  The other non-optional parameters are
#         self-explanatory.  If ``test_data`` is provided then the
#         network will be evaluated against the test data after each
#         epoch, and partial progress printed out.  This is useful for
#         tracking progress, but slows things down substantially."""
#         print(training_data)
#         if test_data: n_test = len(test_data)
#         n = len(training_data)
#         for j in xrange(epochs):
            
#             mini_batches = [
#                 training_data[k:k+mini_batch_size]
#                 for k in xrange(0, n, mini_batch_size)]
#             for mini_batch in mini_batches:
#                 self.update_mini_batch(mini_batch, eta)
#             # if test_data:
#             #     print "Epoch {0}: {1} / {2}".format(
#             #         j, self.evaluate(test_data), n_test)
#             # else:
#             #     print "Epoch {0} complete".format(j)

#     def update_mini_batch(self, mini_batch, eta):
#         """Update the network's weights and biases by applying
#         gradient descent using backpropagation to a single mini batch.
#         The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
#         is the learning rate."""
#         print(mini_batch)
#         nabla_b = [np.zeros(b.shape) for b in self.biases]
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
#         for x, y in mini_batch:
#             delta_nabla_b, delta_nabla_w = self.backprop(x, y)
#             nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
#             nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
#         self.weights = [w-(eta/len(mini_batch))*nw
#                         for w, nw in zip(self.weights, nabla_w)]
#         self.biases = [b-(eta/len(mini_batch))*nb
#                        for b, nb in zip(self.biases, nabla_b)]
                


#     def  training(self,x,y,eta):
#         nabla_b = [np.zeros(b.shape) for b in self.biases]
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
#         delta_nabla_b, delta_nabla_w = self.backprop(x, y)
#         nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
#         nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
#         self.weights = [w-(eta/len(mini_batch))*nw
#                         for w, nw in zip(self.weights, nabla_w)]
#         self.biases = [b-(eta/len(mini_batch))*nb
#                        for b, nb in zip(self.biases, nabla_b)]

#     def backprop(self, x, y):
#         """Return a tuple ``(nabla_b, nabla_w)`` representing the
#         gradient for the cost function C_x.  ``nabla_b`` and
#         ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
#         to ``self.biases`` and ``self.weights``."""
#         nabla_b = [np.zeros(b.shape) for b in self.biases]
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
#         # feedforward
#         activation = x
#         activations = [x] # list to store all the activations, layer by layer

#         zs = [] # list to store all the z vectors, layer by layer
#         for b, w in zip(self.biases, self.weights):
#             z = np.dot(w, activation)+b
#             zs.append(z)
#             activation = sigmoid(z)
#             activations.append(activation)
#         # backward pass
#         delta = self.cost_derivative(activations[-1], y) * \
#             sigmoid_prime(zs[-1])
#         nabla_b[-1] = delta
#         nabla_w[-1] = np.dot(delta, activations[-2].transpose())
#         # Note that the variable l in the loop below is used a little
#         # differently to the notation in Chapter 2 of the book.  Here,
#         # l = 1 means the last layer of neurons, l = 2 is the
#         # second-last layer, and so on.  It's a renumbering of the
#         # scheme in the book, used here to take advantage of the fact
#         # that Python can use negative indices in lists.
#         for l in xrange(2, self.num_layers):
#             z = zs[-l]
#             sp = sigmoid_prime(z)

#             delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
#             nabla_b[-l] = delta
#             activations[-l-1] = np.matrix(activations[-l-1])
#             nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
#         return (nabla_b, nabla_-20
    
#     def cost_derivative(self, output_activations, y):
#         """Return the vector of partial derivatives \partial C_x /
#         \partial a for the output activations."""
#         return (output_activations-y)


#          #### Miscellaneous functions
 

net = NeuralNetwork([2,6,1])            
def main():     
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )
      
        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)
        showGameOverScreen(crashInfo)

        


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    while True:
        # for event in pygame.event.get():
        #     if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        #         pygame.quit()
        #         sys.exit()
        #     #if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                # make first flap sound and return values for mainGame
      #  SOUNDS['wing'].play()
        return {
            'playery': playery + playerShmVals['val'],
            'basex': basex,
            'playerIndexGen': playerIndexGen,
            }

        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 5 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def mainGame(movementInfo):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()
    print("new pipe",newPipe1)
    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerRot     =  45   # player's rotation
    playerVelRot  =   3   # angular speed
    playerRotThr  =  20   # rotation threshold
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    # Defining input for feedforward
    global X
    
    while True:
        # for event in pygame.event.get():
        #     if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        #         pygame.quit()
        #         sys.exit()
        #     if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
        #         if playery > -2 * IMAGES['player'][0].get_height():
        #             # playerVelY = playerFlapAcc
        #             # playerFlapped = True
        #             # SOUNDS['wing'].play()
        # if(len(lowerPipes)==3):

        #     a = [lowerPipes[1]['x']+4-playerx,(lowerPipes[1]['y'])-playery]

        #     X = np.array([[lowerPipes[1]['x']+4-playerx,lowerPipes[1]['y']-20-playery]])

        # elif (len(lowerPipes)==2 and lowerPipes[0]['x']<50):
        #     a = [lowerPipes[1]['x']-playerx,lowerPipes[1]['y']-playery]
        #     X = np.array([[lowerPipes[1]['x']+4-playerx,lowerPipes[1]['y']-20-playery]])    
        # else:
        #     a = [lowerPipes[0]['x']-playerx,lowerPipes[0]['y']-playery]
        #     X = np.array([[lowerPipes[0]['x']+4-playerx,lowerPipes[0]['y']-20-playery]])

        if(lowerPipes[0]['x'] < 57):
            x_data = lowerPipes[1]['x']-playerx;
            y_data = lowerPipes[1]['y']-playery;                    
        else:
            x_data = lowerPipes[0]['x']-playerx;
            y_data = lowerPipes[0]['y']-playery;
        X = np.array([[x_data, y_data]])
        #crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               #upperPipes, lowerPipes)
        # print(lowerPipes)                       
        # print(playerx,playery)
        

        for e in X:
            output = net.predict(e)  
        # print(output)   
        if(output>0.5):
            playerVelY = playerFlapAcc
            playerFlapped = True
            #SOUNDS['wing'].play()

        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        # print(crashTest[0])
        if(playery<0 or crashTest[0]==True):
            # print('Hello')
            if(output>0.5):    
                y = np.array([0])
            else:
                y = np.array([1])    
            # print("Fitting")
            # net.fit(X,y)
        else:
            if(output>0.5):    
                y = np.array([1])
            else:
                y = np.array([0]) 
            # net.fit(X,y)  
        # check for crash here
        # if(len(lowerPipes)==3):
        #     a = [lowerPipes[1]['x']-playerx,lowerPipes[1]['y']-playery]
        #     X = np.array([[lowerPipes[1]['x']-playerx,lowerPipes[1]['y']-playery]])
    
        # else:
        #     a = [lowerPipes[0]['x']-playerx,lowerPipes[0]['y']-playery]
        #     X = np.array([[lowerPipes[0]['x']-playerx,lowerPipes[0]['y']-playery]])

        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        

        if crashTest[0] or playery<0:

            print ("----------------------------------------------------------------")
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
                'playerRot': playerRot
            }
        global max_score
        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                max_score = max(max_score,score)
                # print(max_score)
                #SOUNDS['point'].play()

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # rotate the player
        if playerRot > -90:
            playerRot -= playerVelRot

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            playerRot = 45

        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)

        # Player rotation has a threshold
        visibleRot = playerRotThr
        if playerRot <= playerRotThr:
            visibleRot = playerRot
        
        playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (playerx, playery))

        pygame.display.update()
        FPSCLOCK.tick(FPS)



def showGameOverScreen(crashInfo):
    """crashes the player down ans shows gameover image"""
    score = crashInfo['score']
    playerx = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    playerRot = crashInfo['playerRot']
    playerVelRot = 7

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    # play hit and die sounds
    
    if not crashInfo['groundCrash']:
        print('hell0')


    while True:
        # for event in pygame.event.get():
        #     if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        #         pygame.quit()
        #         sys.exit()
        #     if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
        #         if playery + playerHeight >= BASEY - 1:
        theta1 = net.weights[0].tolist()
        theta2 = net.weights[1].tolist()
        dict_obj = {"theta1" : theta1, "theta2" : theta2}
        with open('weights.json', 'w') as outfile:
            json.dump(dict_obj, outfile)

        print(net.weights)
        return

        # player y shift
        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # rotate only when it's a pipe crash
        if not crashInfo['groundCrash']:
            if playerRot > -90:
                playerRot -= playerVelRot

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)

        playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
        SCREEN.blit(playerSurface, (playerx,playery))

        FPSCLOCK.tick(FPS)
        pygame.display.update()



def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == '__main__':
    main()
