import gym
from gym.envs.registration import register
from gym.envs.toy_text.frozen_lake import generate_random_map
import readchar
import colorama as cr

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT
}

# Register FrozenLake with is_slippery False
cr.init(autoreset=True)
register(
    id='FrozenLake-v3',
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make("FrozenLake-v3", render_mode='human')
env.reset()
env.render()

while True:
    # Choose an action from keyboard
    key = readchar.readkey()

    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    print(arrow_keys[key])
    state, reward, done, info1, info = env.step(action)
    env.render()  # Show the board after action
    print("State:", state, "Action", action, "Reward:", reward, "info:", info)

    if done:
        print("Finished with reward", reward)
        break