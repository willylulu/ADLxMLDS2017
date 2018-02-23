import os
from agent_dir.agent import Agent
import numpy as np
import scipy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.batch_size = 1024
        self.action_size = 3
        
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
            self.model.load_weights("pong.h5")

        ##################
        # YOUR CODE HERE #
        ##################
        
    def preprocess(self, o):
        y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, [80,80])
        return np.expand_dims(resized.astype(np.float32),axis=2)

    def getModel(self):
        
        model = Sequential()
        model.add(Conv2D(32, (8,8), strides=(4,4), padding="same"
                         , activation='relu', kernel_initializer='truncated_normal', input_shape=(80, 80, 1)))
        model.add(Conv2D(64, (4,4), strides=(2,2), padding="same"
                         , kernel_initializer='truncated_normal', activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        

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
    
    def trainModel(self, ite):
        gradients = np.vstack(self.gradients[:ite])
        rewards = np.vstack(self.rewards[:ite])
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        gradients *= rewards
        X = np.vstack([self.states[:ite]])
        Y = self.probs[:ite] + self.learning_rate * np.squeeze(np.vstack([gradients]))
        i = 0
        total_loss = 0
#         while i<len(X):
#             loss = self.model.train_on_batch(X[i:(i+self.batch_size)], Y[i:(i+self.batch_size)])
#             total_loss += loss
#             i += self.batch_size
#         loss = self.model.train_on_batch(X, Y)
#        self.model.fit(X, Y, batch_size=self.batch_size, epochs=1)
        try:
            total_loss = self.model.train_on_batch(X, Y)
        except:
            print("Fuck me!")
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
        self.flag = len(os.listdir("./record"))
        if os.path.exists("pong.h5"):
            print("got model!")
            self.model.load_weights("pong.h5")
            
        self.states = [None]*10000
        self.gradients = [None]*10000
        self.rewards = [None]*10000
        self.probs = [None]*10000
        
        self.record = []
        
        for i in range(100000):
            
            state = self.env.reset()
            score = 0
            done = False
            prev_x = None
            ite = 0
            
            while done!=True:
                cur_x = self.preprocess(state)
                x = cur_x - prev_x if prev_x is not None else np.zeros([80, 80, 1])
                prev_x = cur_x
                
                self.states[ite] = x
                
                action, prob = self.getAction(x)
                
                state, reward, done, info = self.env.step(action)
                score += reward
                
                y = np.zeros([self.action_size])
                y[action-1] = 1
                
                self.gradients[ite] = np.array(y).astype('float32') - prob
                self.rewards[ite] = reward
                self.probs[ite] = prob
                ite += 1
            
            Xlen, loss = self.trainModel(ite)
            print('Episode: %4d / Step size: %d / Score: %f / Loss: %f.' % (i, Xlen, score, loss))
            self.record.append(score)
            if i>0 and i%10==0:
                np.save("./record/"+str(self.flag)+".npy",np.array(self.record))
                self.record = []
                self.flag += 1
                self.model.save_weights("pong.h5")
            
    def getAction(self, observation):
        x = np.expand_dims(observation, axis=0)
        prob = self.model.predict(x, batch_size=1)
        prob = prob[0]
        prob = prob / np.sum(prob)
        action = np.argmax(prob)
        
        return action+1, prob  

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

