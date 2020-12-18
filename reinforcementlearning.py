
import gym

env = gym.make("Taxi-v3").env

env.render()

env.reset()

env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
state= env.encode(3, 1, 2, 0)
print("State:", state)
env.s = state
env.render()

env.P[328]

env.s = 328
epochs = 0
penalties, rewards = 0, 0
frames = []
done = False
while not done:
  action = env.action_space.sample()
  state, reward, done , info = env.step(action)

  if reward == -10:
     penalties += 1

  frames.append({
      'frame': env.render(mode='ansi'),
      'state': state,
      'action': action,
      'reward': reward
       }
  )

  epochs += 1

  print("Timesteps taken: {}".format(epochs))
  print("Penalties incureed: {}".format(penalties))

from IPython.display import clear_output
from time import sleep

def print_frames(frames):
  for i, frame in enumerate(frames):
    clear_output(wait=True)
    print(frame['frame'])
    print(f"Timestep: {i + 1}")
    print(f"State: {frame['state']}")
    print(f"Action: {frame['action']}")
    print(f"Reward: {frame['reward']}")
    sleep(.1)
print_frames(frames)

import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])

%%time
import random
from IPython.display import clear_output

  alpha = 0.1
  gamma = 0.6
  epilson = 0.1
 
  all_epochs = []
  all_penalties = []
 
  for i in range(1, 100001):
      state = env.reset()

   epochs, penalties, reward, = 0, 0, 0

    done = False
    while not done:
        if random.uniform(0, 1) < epilson:
            action = env.action_space.sample()
         else:
            action = np.argmax(q_table[state])
 
         next_state, reward, done, info = env.step(action)
         old_value = q_table[state, action]
         next_max = np.max(q_table[next_state])
         new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
         q_table[state, action] = new_value
         if reward == -10:
         penalties += 1

         state = next_state
        epochs += 1
      
    if i % 100 == 0:
         clear_output(wait=True)
         print(f"Episode: {i}")
 
 print("Training finshed.\n")

q_table[328]

total_epochs, total_penalties = 1, 0
episodes = 100

for _ in range(episodes):
  state = env.reset()
  epochs, penalties, reward = 1, 0, 0

  done = False

  while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)

    if reward == -10:
      penalties += 1

      epochs +=1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episodes: {total_penalties/ episodes}")







