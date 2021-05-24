import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import math
import numpy as np
import random
from collections import deque
import pandas as pd


class Agent:

    def __init__(self, fork, action_size):
        #buy/sell - action size
        self.state_size  = fork
        self.action_size = action_size 
        self.model = self.create_model()
        self.memory_to_learn = deque(maxlen = 64)
        self.batch_size = 16

        #how to value future
        self.gamma = 0.95
        self.alpha = 0.85

        #epsilon-greedy action policy
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear")) 
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def enrich(self,state):
        state_normalized = state # if we want to normalize, do it here
        state_normalized = type_to_NN(state_normalized) #NN input is (batch,state) and its numpy array, (look self.model.summary())

        #Goes under exploitation and exploitation dillema
        if(random.uniform(0, 1) <= 1-self.epsilon):
            return np.argmax(self.model.predict(state_normalized)[0])
        else:
            return random.randrange(self.action_size)

    def add_experience(self,exp):
        self.memory_to_learn.append(exp)

    def learn(self):
        #print("Actual epsilon: {}".format(self.epsilon))
        batch = []
        if self.get_mem_size() <= self.batch_size:
            batch = random.sample(self.memory_to_learn, self.get_mem_size())
        else: batch = random.sample(self.memory_to_learn, self.batch_size)

        for state, action, reward, next_state in batch:
            Q_old = self.model.predict(type_to_NN(state))
            max_Q = self.model.predict(type_to_NN(next_state))
            new_Q_value = (1-self.alpha) * Q_old[0][action] + self.alpha*(reward+ self.gamma * np.amax(max_Q[0]))
            Q_old[0][action] = new_Q_value
            self.model.fit(type_to_NN(state), Q_old, epochs=1, verbose=0)
    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_mem_size(self):
        i = 0
        for el in self.memory_to_learn:
            i +=1
        return i


class Market:
    def __init__(self,name, fork, column_name = "Close"):
        self.data = read_csv(name, column_name)
        self.data_length = len(self.data)
        self.timestamp = 0
        self.fork = fork + 1
        self.current_prices = self.data[:self.fork]

    def play_on_market(self, action, stocks_amount):
        done = False
        if self.timestamp + self.fork + 1 == self.data_length: done = True
        diff = self.data[self.timestamp + self.fork]-self.data[self.timestamp + self.fork-1]
        reward = 0
        self.timestamp += 1
        self.current_prices = self.data[self.timestamp : self.timestamp + self.fork]
        if action == 0: reward = stocks_amount*diff 
        elif action == 1: reward = -stocks_amount*diff 
        return (self.current_prices, reward, done)

    def get_current_prices(self): return self.current_prices

    def reset_market(self):
        self.timestamp = 0
        self.current_prices = self.data[:self.fork]


def read_csv(name,column_name):
    data = pd.read_csv(name+".csv")
    data_column = [data.at[i,column_name] for i in range(len(data))]
    return data_column

def count_state(prices):
   diffrences = [prices[i+1]-prices[i] for i in range(len(prices)-1)]
   return diffrences

def type_to_NN(X):
    return np.asarray([X])
  
  
fork  = 5
action_size = 2
episodes = 5
stocks_amount = 2
agent  = Agent(fork,action_size)
market = Market("TSLA",fork)

for i in range(episodes):
    print("\n\n----------STARTING EPISODE {}.-----------\n\n".format(i))
    market.reset_market()
    state = count_state(market.get_current_prices())
    done = False
    summary_reward = 0
    while not done:
        action = agent.enrich(state)
        #print("Action = {}".format(action))
        new_prices, reward, done = market.play_on_market(action,stocks_amount)
        new_state = count_state(new_prices)
        summary_reward += reward
        #print("New prices: {}\nReward: {}\nDone: {}.".format(new_prices,reward,done))
        agent.add_experience((state, action, reward, new_state))
        #print("Actual profit/loss: {}.\n".format(summary_reward))
        agent.learn()
        state = new_state
    
    print("\nEpisode {} profit/loss with: {}.\n".format(i, summary_reward))
