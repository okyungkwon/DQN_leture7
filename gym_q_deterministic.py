import gym
import readchar

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {'\x1b[A': UP,
              '\x1b[B': DOWN,
              '\x1b[C': RIGHT,
              '\x1b[D': LEFT}

# is_slippery True(Default)
env = gym.make('FrozenLake-v1', render_mode="human")

env.reset()

# print_utils.clear_screen()
env.render() # show the initial board

while True:
    # Choose an action from keyboard
    key = readchar.readkey()

    if key not in arrow_keys.keys():
        print('Game aborted!')
        break

    action = arrow_keys[key]
    state, reward, done, _, _ = env.step(action)

    # show the board after action
    env.render()

    print("state: {} Action: {} Reward: {}".format(state, action, reward))

    if done:
        print("Game Over")
        print("Finished with reward: ", reward)
        break
