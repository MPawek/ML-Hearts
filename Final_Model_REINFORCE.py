
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Import pettingZoo env that wraps HeartsEngine
from hearts_env import env 

# Global Variables
# Observation length = 1 (Play/Pass) + 1 (hearts broken) + 4 (trick suit, 1 of 4 possible)
# + 1 (first trick flag) + 52 (cards in hand) + 52 (cards played in current trick) + 4 (normalized round scores) = 115
OBS_DIM = 115           # Observation dimension: total length of obs vector
NUM_CARDS = 52          # Number of cards possible in deck, also number of output neurons
GAMMA = 0.99            # Discount factor - Hyperparameter, may change

# Policy model is shared by all 4 players, so must be global
policy_model = tf.keras.Sequential([
    layers.Input(shape=(OBS_DIM,)),                    # observation vector
    layers.Dense(128, activation="relu"),              # hidden layer 1
    layers.Dense(128, activation="relu"),              # hidden layer 2
    layers.Dense(NUM_CARDS, activation="softmax"),     # π(a|s)
    ])
        
# Optimizer = Nadam (Hyperparameter) with learning_rate 0.001 (Hyperparameter)
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)

# Function to determine action based on policy and action mask
def modelPolicy(obs_vector, action_mask):
    # Have to add batch dimension for Keras, caused problems otherwise
    obs_batch = np.expand_dims(obs_vector, axis=0)

    # Create array of size 52 to hold model's output softmax probabilities of playing each card
    probabilities = policy_model(obs_batch, training=False).numpy()[0] 

    # We have to mask out illegal actions, in action_mask illegal actions are 0 and legal actions are 1 so we can multiply to get rid of them
    masked_probabilities = probabilities * action_mask

    # We need to renormalize probabilities after removing illegal actions, so we find the sum of the remaining probabilities
    total = masked_probabilities.sum()

    # We have to put a check for non legal moves for debugging purposes:
    # If no legal moves:
    if total == 0:
        print("  WARNING: no legal actions available, passing None")
        return None

    # We then divide the probabilities by the total to find the new probability percentages
    masked_probabilities /= total

    # Return a random card choice from the present probabilities -- this is the experimentation of the model
    action = np.random.choice(NUM_CARDS, p=masked_probabilities)

    # Return the card choice - cast to int to make sure it will work with the array
    return int(action)

# For REINFORCE Policy Gradient Algorithm we have to discount rewards
def discount_rewards(rewards, gamma=GAMMA):
    # Rewards have to be set to float for the math we will do to discount
    rewards = np.array(rewards, dtype=np.float32)
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)

    # Initialize total variable
    total = 0.0

    # Going backwards, add to our rewards the gamma * currentTotal so future rewards are discounted
    for index in reversed(range(len(rewards))):
        total = rewards[index] + gamma * total
        discounted_rewards[index] = total

    # Return the discounted rewards
    return discounted_rewards

# Function to play single hand/round of Hearts
def play_hand(hearts_env):
    # Set Trajectories as variable that holds all important lists:
    trajectories = {"obs": [], "actions": [], "rewards": []}

    # Reset the environment for the new round
    hearts_env.reset()

    # We have to track rewards from last action taken per player for that specific environment
    # Initialize that saved information here
    last_obs = {agent: None for agent in hearts_env.possible_agents}
    last_action = {agent: None for agent in hearts_env.possible_agents}

    # PettingZoo will loop over all agents in turn order until the hand is over
    for agent in hearts_env.agent_iter():
            # Get environmental data from agent whose turn it is
            obs, reward, terminated, truncated, info = hearts_env.last()

            # Print basic debug info
            # print(f"Step {step_count:3d} | Agent: {agent:8s} | "
            #     f"Reward: {reward:5.2f} | Terminated: {terminated} | Truncated: {truncated}")

            # We want to save the reward for the agent's previous action
            # This is applied later due to possibility of trick not having ended yet
            # If there is a previous action/environment:
            if last_obs[agent] is not None and last_action[agent] is not None:
                trajectories["obs"].append(last_obs[agent])
                trajectories["actions"].append(last_action[agent])
                trajectories["rewards"].append(float(reward))


            # If this agent is done for this hand → no action
            if terminated or truncated:
                last_obs[agent] = None
                last_action[agent] = None
                hearts_env.step(None)
                continue

            # Get observations
            obs_vector = obs["observation"]
            # Get action mask
            action_mask = obs["action_mask"]

            # Call the model and get the policy choice:
            action = modelPolicy(obs_vector, action_mask)

            # Save the observations and action to the last turn variables
            last_obs[agent] = obs_vector
            last_action[agent] = action
 

            # Send chosen action to env to take
            hearts_env.step(action)

    # Return the full set of trajectories for the current hand
    return trajectories

# The function where the model learns using REINFORCE updates from trajectory data
def update_policy(trajectories):
    # Separate the trajectory data
    obs = np.array(trajectories["obs"], dtype=np.float32)
    actions = np.array(trajectories["actions"], dtype=np.int32)
    rewards = np.array(trajectories["rewards"], dtype=np.float32)

    # Call our discount function to modify the rewards
    discounted_rewards = discount_rewards(rewards, gamma=GAMMA)
    # Normalize the returning rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)


    with tf.GradientTape() as tape:
        # Forward pass: Policy_model outputs probabilities for all 52 cards
        probs = policy_model(obs, training=True)
        # Pick out probability of chosen action
        action_one_hot = tf.one_hot(actions, NUM_CARDS)
        # Find probability of chosen actions at each time step
        chosen_action_probs = tf.reduce_sum(probs * action_one_hot, axis=1)
        # Convert the probabilities with log values for loss calculations
        log_probs = tf.math.log(chosen_action_probs + 1e-8)
        # Calculate loss using REINFORCE method
        loss = -tf.reduce_mean(log_probs * discounted_rewards)

    # Now that we have computed loss, find and apply the gradients
    grads = tape.gradient(loss, policy_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

    # Return stats in case we need them
    return float(loss.numpy()), float(rewards.sum()), float(discounted_rewards.sum())
    
# The outside loop that runs the training
# Current number of episodes (rounds) set to 1000
def run_model(num_episodes=1000):
    hearts_env = env()  # creates the wrapped PettingZoo env (with TerminateIllegalWrapper etc.)
    # Variables to save scoring history
    history_round_scores = []
    history_total_scores = []

    for episode in range(1, num_episodes + 1):
        trajectories = play_hand(hearts_env)
        loss, total_reward, total_discounted_rewards = update_policy(trajectories)

        # Get current scores from the HeartsEngine:
        round_scores = hearts_env.last_round_scores
        total_scores = [hearts_env.game.players[i].score for i in range(4)]

        # Save the current scores
        history_round_scores.append(round_scores)
        history_total_scores.append(total_scores)

        print(f"Episode {episode:4d}")
        print(f"Steps: {len(trajectories['obs']):3d}")
        print(f"Total Reward: {total_reward:6.2f}")
        print(f"Loss: {loss:8.4f}")
        print(f"Round Scores: {round_scores}")
        print(f"Total Score: {total_scores}")
        
    # Save the scores:
    np.save("round_scores.npy", np.array(history_round_scores))
    np.save("total_scores.npy", np.array(history_total_scores))

if __name__ == "__main__":
    run_model()
