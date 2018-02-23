import os
from agent_dir.agent import Agent
import numpy as np
import scipy
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D, Input
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from keras import backend as K

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.action_size = 3
        
        self.critic, self.actor, self.train_net, self.adv = self.getModel()
        
        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),
                               loss=[self.value_loss(), self.policy_loss(self.adv, 0.01)])
        if args.test_pg:
            #you can load your model here
            pass

        ##################
        # YOUR CODE HERE #
        ##################
        
    def preprocess(self, o):
        y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, [80,80])
        return np.expand_dims(resized.astype(np.float32),axis=2)
    
    def policy_loss(self, adventage=0., beta=0.01):

        def loss(y_true, y_pred):
            return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(adventage)) + \
                   beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))

        return loss


    def value_loss(self):

        def loss(y_true, y_pred):
            return 0.5 * K.sum(K.square(y_true - y_pred))

        return loss

    def getModel(self):
        
        state = Input(shape=(80, 80, 1))
        h = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu')(state)
        h = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu')(h)
        h = Flatten()(h)
        h = Dense(256, activation='relu')(h)

        value = Dense(1, activation='linear', name='value')(h)
        policy = Dense(self.action_size, activation='softmax', name='policy')(h)

        value_network = Model(inputs=state, outputs=value)
        policy_network = Model(inputs=state, outputs=policy)

        adventage = Input(shape=(1,))
        train_network = Model(inputs=[state, adventage], outputs=[value, policy])

        return value_network, policy_network, train_network, adventage
    
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
        states = np.array(self.states[:ite])
        actions = np.array(self.actions[:ite])
        rewards = np.array(self.rewards[:ite])
        rewards = self.discount_rewards(rewards)
        
        self.unroll = np.arange(ite)
        
        values, policy = self.train_net.predict([states, self.unroll])
        
        adventage = rewards - values.flatten()
        
        total_loss = self.train_net.train_on_batch([states, adventage], [rewards, actions])
        
        return ite, total_loss
            
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
        if os.path.exists("pong_a2c.h5"):
            print("load model")
            self.train_net.load_weights("pong_a2c.h5")
            
        self.states = [None]*10000
        self.actions = [None]*10000
        self.rewards = [None]*10000
        
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
                
                self.actions[ite] = y
                self.rewards[ite] = reward
                ite += 1
            
            Xlen, loss = self.trainModel(ite)
            print('Episode: %4d / Step size: %d / Score: %f / c loss: %f / a loss: %f.' % (i, Xlen, score, loss[1], loss[2]))
            self.record.append(score)
            if i>0 and i%10==0:
                self.train_net.save_weights("pong_a2c.h5")
                np.save("./record/"+str(self.flag)+".npy",np.array(self.record))
                self.record = []
                self.flag += 1
            
    def getAction(self, observation):
        x = np.expand_dims(observation, axis=0)
        prob = self.actor.predict(x, batch_size=1)
        prob = prob[0]
        prob = prob / np.sum(prob)
        action = np.random.choice(3 ,1 ,p=prob)[0]
        
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

