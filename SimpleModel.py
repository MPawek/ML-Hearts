# Simple Setup w/o ANN:
# EXAMPLE of simple policy, general instructions:
def basic_policy (obs):
    hand = obs[whatever_index_hand_is_stored_in]
    # If leading card, choose card to play
    # If not leading card, play legal card
    # Add policy to play hearts when possible, avoid taking hearts when possible
    # If last hand and no hearts, play high
    # Otherwise, play low if possible

# Array for scores from all tries
totals = []
# Number of tries
for episode in range(500):
    # Total rewards for current try
    episode_rewards = 0
    # Reset environment between each try
    obs, info = env.reset(seed=episode)

    # Max number of steps per try
    for step in range(200):
        # Action performed is based on basic_policy given current obs
        action = basic_policy(obs)
        # Take step based on action, get new into
        obs, reward, done, truncated, info = env.step(action)
        # Add reward to total rewards for try
        episode_rewards += reward
        # If ended early, finish the try
        if done or truncated:
            break

    # Add final reward try total to array
    totals.append(episode_rewards)
