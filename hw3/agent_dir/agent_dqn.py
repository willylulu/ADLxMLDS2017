import os
from agent_dir.agent import Agent
import numpy as np
import tensorflow as tf
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from keras import backend as K

def loss_function(y, label):
    error = K.abs(y - label)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    return K.mean(0.5*K.square(quadratic_part)+linear_part, axis=-1)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        set_session(session)
        
        super(Agent_DQN,self).__init__(env)
        
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.targetModel = self._build_model()
        self.targetModel.set_weights(self.model.get_weights())
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            if os.path.exists("breakout.h5"):
                print("got model!")
                self.model.load_weights("breakout.h5")
        ##################
        # YOUR CODE HERE #
        ##################

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss=loss_function, optimizer=RMSprop(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, test):
        
        statex = np.expand_dims(state, axis=0)
        if test:
            act_values = self.model.predict(statex)
            return np.argmax(act_values[0])
        else:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict(statex)
            return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        X = []
        Y = []
        
        for state, action, reward, next_state, done in minibatch:
            statex = np.expand_dims(state, axis=0)
            next_statex = np.expand_dims(next_state, axis=0)
            target = self.model.predict(statex)
            
            if done:
                target[0][action] = reward
            else:
                next_action = self.model.predict(next_statex)[0]
                tar_action = self.targetModel.predict(next_statex)[0]
                target[0][action] = reward + self.gamma * tar_action[np.argmax(next_action)]
            
            X.append(statex)
            Y.append(target)
        
        X = np.vstack(X)
        Y = np.vstack(Y)
        loss = self.model.train_on_batch(X, Y)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= 1e-7
        return loss, max(next_action)
    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        if os.path.exists("breakout.h5"):
            print("got model!")
            self.model.load_weights("breakout.h5")
        
        j = 0
        for i in range(100000):
            state = self.env.reset()
            done = False
            k = 0
            losst = 0
            maxqt = 0
            while done!=True:
                action = self.act(state, False)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if len(self.memory) > 2000 and j%4==0:
                    loss, maxQ = self.replay(32)
                    losst += loss
                    maxqt += maxQ
                j += 1
                k += 1
            if i%10 >0 and i>0:
                self.targetModel.set_weights(self.model.get_weights())
                self.model.save_weights("breakout.h5")
            print("ep: %5d / step: %5d / reward: %f / j: %5d / loss: %f / Q: %f"%(i, k, reward, j, losst, maxqt))

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.act(observation, True)

