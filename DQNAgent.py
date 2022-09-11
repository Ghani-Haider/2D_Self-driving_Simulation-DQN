import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, epsilon = 1.0):
        # action space Structure
        #       (Steering Wheel, speed, Break)
        # Range       -1-1       0-1   0-1

        # (left, acc, 20% brake),    (straight, acc, 20% brake), (right, acc, 20% brake),
        # (left, acc, 0% brake),      (straight, acc, 0% brake), (right, acc, 0% brake),
        # (left, decc, 20% brake),  (straight, decc, 20% brake), (right, decc, 20% brake),
        # (left, decc, 0% brake),    (idle, idle, idle),         (right, decc, 0% brake)

        self.action_space    = [(-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
                                (-1, 1,   0), (0, 1,   0), (1, 1,   0),
                                (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
                                (-1, 0,   0), (0, 0,   0), (1, 0,   0)]
        self.memory          = deque(maxlen=5000)
        self.gamma           = 0.95 # discount rate
        self.epsilon         = epsilon # exploration rate
        self.epsilon_min     = 0.1
        self.epsilon_decay   = 0.9999
        self.learning_rate   = 0.001
        self.model           = self.build_model()
        self.target_model    = self.build_model()
        self.update_target_model()

    def build_model(self):
        # CNN architecture for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def action(self, state):
        if np.random.rand() > self.epsilon:
            # get action with max value
            action_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(action_values[0])
        else:
            # get random action
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        
        # get all predicted outputs (actions) on all inputs (states)
        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            # replace taken action's value with the obtained reward
            if done:
                target[action_index] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            # save the input and output values
            train_state.append(state)
            train_target.append(target)
        # train the model on the data (input: states, output: actions)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)
