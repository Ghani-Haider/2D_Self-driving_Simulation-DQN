import cv2
import numpy as np

# obtain grayscale image of the state (colored image)
def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

# move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)    
    return np.transpose(frame_stack, (1, 2, 0))
