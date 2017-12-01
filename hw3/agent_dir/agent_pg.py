import os
from agent_dir.agent import Agent
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.layers.convolutional import Conv2D
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.batch_size = 2048
        
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        
        self.model = self.getModel()
        self.model.summary()

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            print("got model!")
            self.model = load_model("pong.h5")

        ##################
        # YOUR CODE HERE #
        ##################
        
    def preprocess(self, I):
        I = I[35:195]
        I = I[::2, ::2, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        I = np.expand_dims(I, axis=-1)
        return I.astype(np.float)

    def getModel(self):
        
        model = Sequential()
        model.add(Conv2D(32, 6, subsample=(3, 3), padding='same', 
                         activation='relu', kernel_initializer='he_uniform', input_shape=(80, 80, 1)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.env.action_space.n, activation='softmax'))

        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model
    
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    def trainModel(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        gradients *= rewards
        X = np.vstack([self.states])
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        i = 0
        total_loss = 0
#         while i<len(X):
#             loss = self.model.train_on_batch(X[i:(i+self.batch_size)], Y[i:(i+self.batch_size)])
#             total_loss += loss
#             i += self.batch_size
        loss = self.model.train_on_batch(X, Y)
        total_loss += loss
        return len(X), total_loss
            
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_x = None


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if os.path.exists("pong.h5"):
            print("got model!")
            self.model = load_model("pong.h5")
        for i in range(100000):
            
            state = self.env.reset()
            score = 0
            done = False
            prev_x = None
            self.states = []
            self.gradients = []
            self.rewards = []
            self.probs = []
            
            while done!=True:
                cur_x = self.preprocess(state)
                x = cur_x - prev_x if prev_x is not None else np.zeros([80, 80, 1])
                prev_x = cur_x
                
                self.states.append(x)
                
                action, prob = self.getAction(x)
                
                state, reward, done, info = self.env.step(action)
                score += reward
                
                y = np.zeros([self.env.action_space.n])
                y[action] = 1
                
                self.gradients.append(np.array(y).astype('float32') - prob)
                self.rewards.append(reward)
                self.probs.append(prob)
            
            Xlen, loss = self.trainModel()
            print('Episode: %d / Step size: %d / Score: %f / Loss: %f.' % (i, Xlen, score, loss))
            if i>0 and i%10==0:
                self.model.save("pong.h5")
            
    def getAction(self, observation):
        x = np.expand_dims(observation, axis=0)
        prob = self.model.predict(x, batch_size=1)
        prob = prob[0]
        prob = prob / np.sum(prob)
        action = np.random.choice(self.env.action_space.n, 1, p=prob)
        
        return action[0], prob  

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        
        ##################
        # YOUR CODE HERE #
        ##################
        cur_x = self.preprocess(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros([80, 80, 1])
        self.prev_x = cur_x
        
        action,_ = self.getAction(x)
        return action

