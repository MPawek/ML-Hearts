import gymnasium as gym
import tensorflow as tf
import numpy as np

# Make the hearts environment
# Image is returned as NumPy array
env = gym.make("hearts_v1.py", render_mode="rgb_array")

# ANN Policy
# Takes obs as input, then outputs action

# Sequential model is for simple ANN
model = tf.keras.Sequential([
    tf.keras.layers.input(obs_dim)
    # Only 5 layers for simple task, maybe start adding more later
    # Relu for ANN dense layers
    # Number of inputs is the size of obs
    # Our output should be probability of each card, even if only 13 in hand
    # Softmax is used for multiple possible outputs
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(52, activation="softmax"),
])

# REINFORCE Policy Gradient Algorithm
# Let Neural Network policy play the game several times, and at each step compute 
# gradients that would make the chosen action even more likely
# Once several episodes have run, compute each action's advantage, using the above method
# If action advantage is positive, the action was probably good and the earlier computed 
# gradients should be applied so it can be chosen again. The reverse is also true.
# Compute mean of all resulting gradient vectors, and use it to perform a gradient descent step

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:

        # Start by calling model, then giving it single observation
        # It outputs probability of actions
        left_proba = model(obs[np.newaxis])

        # Sample random float between 0 and 1, and check whether it's greater than
        # the probability of the output, allowing for exploration and exploitation
        action = (tf.random.uniform([1, 1]) > left_proba)

        # Target probability of action is computed and defined
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)

        # Loss is computed with our loss function
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))

    # Gradient of loss is computed 
    grads = tape.gradient(loss, model.trainable_variables)

    # Step is played with chosen action, and we get the observation and reward back
    obs, reward, done, truncated, info = env.step(int(action))

    # We return those and the gradients we computed
    return obs, reward, done, truncated, grads

# A function that uses the above function to play multiple episodes
def play_multiple_episodes(env, num_hands, n_max_steps, model, loss_fn):
    # Arrays to hold records for rewards and gradients from multiple episodes
    all_rewards = []
    all_grads = []

    # For each episode:
    for hand in range(num_hands):
        # Arrays to hold records for rewards and gradients of current episode
        current_rewards = []
        current_grads = []

        # Reset environment and get current obs
        obs, info = env.reset()

        # For each agent in current episode
        for agent in env.agent_iter():
            # Observe current state and rewards of environment
            obs, reward, done, truncated, grads = env.last()

            # Play one step, receive obs and reward
            obs, reward, done, truncated, grads = play_one_step(env, obs, model, loss_fn)

            # Append values into current arrays
            current_rewards.append(reward)
            current_grads.append(grads)

            # If episode has ended, break
            if done or truncated:
                break
        
        # Append final arrays of episode into full records arrays
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)

    # Return records arrays
    return all_rewards, all_grads


# This will play the game several times, then go back and look at the rewards, discount them, and normalize them

# First, we compute the sum of future discounted rewards at each step
def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

# Then the second will normalize all the discounted rewards across episodes
def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

# Hyperparameters for this would also be defined:
# Number of training iterations:
n_iterations = 150

# Number of episodes per iteration:
n_episodes_per_update = 10

# Max number of steps per episode:
n_max_steps = 200

# Discount factor for rewards:
discount_factor = 0.95

# Optimizer and loss function
# I'll have to switch the loss function to one that works with multiple outputs
# Read back up on optimizers to see which would be good
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = tf.keras.losses.binary_crossentropy

# At each iteration, multiple episodes are played and rewards and gradients are returned
for iteration in range(n_iterations):
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn
        )
    
    # Discount and normalize to find out how good or bad each action was
    all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
    all_mean_grads = []

    # Go through each trainable variable and compute the weighted mean of gradients for that 
    # variable over all the episodes and steps
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)

    # These computed gradients are applied to the optimizer
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))



# Initialize the environment using reset()
# Observation (obs) depends on environment, likely just 13 for hand of cards? Maybe 15 for number of hearts taken and if QoS has been taken?
# 18 for cards played earlier in trick. Maybe just 4 if one is a size-52 array to keep track of what cards are in hand? 5 if it also includes cards seen?
# Reset returns dictionary with environment information, scores?

# Seed is for random order of deck? Or just shuffle instead?
obs, info = env.reset(seed=1)
# obs stores observations of environment in array
# Can use obs[index] to determine values stored in observations, likely used as hand of cards

# Renders the environment in image. Maybe do simple representation of cards?
img = env.render()
# imshow() displays image

# Asks environment what actions are possible, likely to be max 13 options for cards to play, but may have to add 3 more for choices to swap
env.action_space

# This will tell the model to play the 13th card
action = 13
# This will tell the environment what action the actor would like to take, and it returns the new info
obs, reward, done, truncated, info = env.step(action)
# obs: new observation data
# reward: going to be the amount of points taken, which will be negative
# done: boolean variable that marks when episode is over, and environment has to be reset. Have to determine if this will be for tricks or entire game
# truncated: boolean variable marked true when episode is interrupted early (e.g. max number of allowed steps - 13 rounds?)
# info: environment-specific dictionary with extra information, like scores

# call close() after environment is finished