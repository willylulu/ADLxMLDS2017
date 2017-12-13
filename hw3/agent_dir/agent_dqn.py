import os
from agent_dir.agent import Agent
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D, Lambda
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
        
        self.dqn_double = True
        self.dqn_duel = True
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        set_session(session)
        
        super(Agent_DQN,self).__init__(env)
        
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_step = 100000
        self.epsilon_decay = (self.epsilon - self.epsilon_min)/self.epsilon_step
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.targetModel = self._build_model()
        self.targetModel.set_weights(self.model.get_weights())
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            if os.path.exists("breakout-Copy1.h5"):
                print("got Copy model!")
                self.model.load_weights("breakout-Copy1.h5")
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
        if not self.dqn_duel:
            model.add(Dense(self.env.action_space.n))
        else:
        
            model.add(Dense(self.env.action_space.n + 1))
            model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), 
                             output_shape=(self.action_size,)))
        model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, test):
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        
        statex = np.expand_dims(state, axis=0)
        if test:
#             if np.random.rand() <= 0.001:
#                 return random.randrange(self.action_size)
            act_values = self.model.predict(statex)
            return np.argmax(act_values[0])
        else:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict(statex)
            return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
#         X = []
#         Y = []
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        targets = self.model.predict(states)
        next_actions = self.model.predict(next_states)
        target_actions = self.targetModel.predict(next_states)
        
        temp = np.argmax(next_actions, axis=-1)
#         dqn
        if not self.dqn_double:
            targets[range(batch_size),actions] = rewards + (1 - dones) * self.gamma * np.max(target_actions, axis=1)
        else:
            targets[range(batch_size),actions] = rewards + (1 - dones) * self.gamma * target_actions[range(batch_size),temp]
        
#         for state, action, reward, next_state, done in minibatch:
#             statex = np.expand_dims(state, axis=0)
#             next_statex = np.expand_dims(next_state, axis=0)
#             target = self.model.predict(statex)
            
#             if done:
#                 target[0][action] = reward
#             else:
#                 next_action = self.model.predict(next_statex)[0]
#                 tar_action = self.targetModel.predict(next_statex)[0]
#                 target[0][action] = reward + self.gamma * tar_action[np.argmax(next_action)]
            
#             X.append(statex)
#             Y.append(target)
        
#         X = np.vstack(X)
#         Y = np.vstack(Y)
        loss = self.model.train_on_batch(states, targets)
        return loss, np.amax(next_actions)
    
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
        self.flag = len(os.listdir("./record2"))
        if os.path.exists("breakout.h5"):
            print("got model!")
            self.model.load_weights("breakout.h5")
            
        if os.path.exists("memory.pickle"):
            print("got memory!")
            with open("memory.pickle", "rb") as f:
                temp = pickle.load(f)
                print(len(temp))
                for x in temp:
                    self.memory.append(x)
            
        j = 0
        self.record = []
        for i in range(100000):
            state = self.env.reset()
            done = False
            k = 0
            losst = 0
            score = 0
            maxqt = []
            while done!=True:ㄐ
                action = self.act(state, False)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                loss = 0
                maxQ = 0
                if len(self.memory) > 2000 and j%4==0:
                    loss, maxQ = self.replay(32)
                    losst += loss
                    maxqt.append(maxQ)
                if j%1000==0 and j > 0:
                    self.targetModel.set_weights(self.model.get_weights())
                j += 1
                k += 1
                
            if len(maxqt) > 0:
                self.record.append([score, np.max(maxqt)])
                print("ep: %5d / step: %5d / reward: %f / j: %5d / loss: %f / Q: %f"%(i, k, score, j, losst/k, np.max(maxqt)))
            
            if i%100 == 0 and i>0 :
                self.model.save_weights("breakout.h5")
                with open("memory.pickle", "wb") as f:
                    pickle.dump(list(self.memory), f)
                np.save("./record2/"+str(self.flag)+".npy",np.array(self.record))
                self.record = []
                self.flag += 1

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

