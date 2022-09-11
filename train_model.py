import gym
from collections import deque
from DQNAgent import DQNAgent
from helper_functions import *

ENDING_EPISODE                = 300 # total training sessions
SKIP_FRAMES                   = 2 # total frame for each state
TRAINING_BATCH_SIZE           = 64
UPDATE_TARGET_MODEL_FREQUENCY = 5

def main():
    # create gym environment and agent
    env = gym.make('CarRacing-v1')
    agent = DQNAgent()

    # train the agent
    for episode in range(1, ENDING_EPISODE+1):
        # initializing values for each episode
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state] * 3, maxlen=3)
        time_frame_counter = 1
        done = False
        
        while True:
            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.action(current_state_frame_stack)

            reward = 0
            # get reward for state's each frame
            for each_frame in range(SKIP_FRAMES+1):
                next_state, frame_reward, done, info = env.step(action)
                reward += frame_reward
                if done:
                    break

            # terminate this episode if continously getting negative rewards
            if time_frame_counter > 100 and reward < 0:
                negative_reward_counter = negative_reward_counter + 1
            else:
                negative_reward_counter = 0

            # extra bonus if the model uses full gas
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            # save next state
            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            # save the agent's current state, action, reward and next state
            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print('Episode: {}/{}, Time Frames: {}, Total Rewards(adjusted): {:.2}'.format(episode, ENDING_EPISODE, time_frame_counter, float(total_reward)))
                break
            
            # allow the model to learn on saved states
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.learn(TRAINING_BATCH_SIZE)

            time_frame_counter += 1

        if episode % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if episode == ENDING_EPISODE:
            agent.save('./save/trial_{}.h5'.format(episode))

    env.close()

if __name__ == '__main__':
    main()