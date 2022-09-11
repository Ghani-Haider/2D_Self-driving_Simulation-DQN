import gym
from collections import deque
from DQNAgent import DQNAgent
from helper_functions import *

# Group Members
# Ghani Haider gh05177
# Sarrah       sh04400

# load the save models
MODEL = 'saved_models\episode_500.h5'

def main():
    # create gym environment and agent
    env = gym.make('CarRacing-v1')
    agent = DQNAgent(epsilon=0) # epsilon 0 allows all actions to be instructed by the agent
    agent.load(MODEL)

    # initializing values
    init_state = env.reset()
    init_state = process_state_image(init_state)

    total_reward = 0
    state_frame_stack_queue = deque([init_state] * 3, maxlen=3)
    time_frame_counter = 1
    
    # play the game
    while True:
        env.render()

        current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
        # take an action
        action = agent.action(current_state_frame_stack)
        next_state, reward, done, info = env.step(action)

        total_reward += reward

        # save next state
        next_state = process_state_image(next_state)
        state_frame_stack_queue.append(next_state)

        if done:
            print('Time Frames: {}, Total Rewards: {:.2}'.format(time_frame_counter, float(total_reward)))
            break
        time_frame_counter += 1
    
    env.close()

if __name__ == '__main__':
    main()