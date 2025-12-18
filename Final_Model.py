
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from Card import Card, Suit, Rank

# Import pettingZoo env that wraps HeartsEngine
from hearts_env import env 

# Global Variables
# Observation length = 1 (Play/Pass) + 1 (hearts broken) + 4 (trick suit, 1 of 4 possible)
# + 1 (first trick flag) + 52 (cards in hand) + 52 (cards played in current trick) + 4 (normalized round scores) = 115
OBS_DIM = 115           # Observation dimension: total length of obs vector
NUM_CARDS = 52          # Number of cards possible in deck, also number of output neurons
GAMMA = 0.99            # Discount factor - Hyperparameter, may change
BATCH_SIZE = 5          # Implement learning in batches instead of per-hand

# For training a single agent against other trained agents, we choose player 1 (or player_0 in the env)
LEARNING_AGENT = None
# We will also add the ability for a human to play as well
HUMAN_AGENT = "player_2"
# For validation we'll use our trained agent against random agents
VAL_AGENT = "player_0"

# To keep training results, we need to save the weights
SAVE_PATH_POLICY = "policy.weights.h5"
SAVE_PATH_VALUE = "value.weights.h5"

# Policy model is shared by all 4 players, so must be global
# Sequential model, with 3 hidden layers
# 256 was found to be a good amount of neurons
# Relu used for ease of calcuations
# Softmax used due to many potential plays returned
policy_model = tf.keras.Sequential([
    layers.Input(shape=(OBS_DIM,)),                    # observation vector
    layers.Dense(256, activation="relu"),              # hidden layer 1
    layers.Dense(256, activation="relu"),              # hidden layer 2
    layers.Dense(256, activation="relu"),              # hidden layer 3
    layers.Dense(NUM_CARDS, activation="softmax"),     # π(a|s)
    ])
        
# Optimizer = Nadam (Hyperparameter) with learning_rate 0.0001 (Hyperparameter)
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001)

# Actor-Critic Framework (REINFORCE not working by itself)
# Almost same setup as policy model, but only 1 output
value_model = tf.keras.Sequential([
    layers.Input(shape=(OBS_DIM,)),                    # observation vector
    layers.Dense(256, activation="relu"),              # hidden layer 1
    layers.Dense(256, activation="relu"),              # hidden layer 2
    layers.Dense(256, activation="relu"),              # hidden layer 3
    layers.Dense(1, activation=None),                    
])

# Weight for value loss in total loss
VALUE_LOSS_COEFF = 0.5  

# Help to visualize cards for human player:
def index_to_card_string(card_index: int) -> str:
    suit_id = card_index // 13
    rank_id = (card_index % 13) + 2

    card = Card(rank_id, suit_id)
    return str(card)

# Determines human player's current seat
def get_human_index(hearts_env):
    return hearts_env.agentPlayerIndex(HUMAN_AGENT)

# Help to visualize human player's hand
def print_full_hand(hearts_env, human_index: int):
    hand_mask = hearts_env.game.players[human_index].hand.hand
    cards = [index_to_card_string(i) for i, v in enumerate(hand_mask) if v == 1]

    print("\nYour full hand:")
    if cards:
        def suit_key(card_str: str):
            return card_str[-1]  # 'c','d','s','h'
        cards_sorted = sorted(cards, key=suit_key)
        print("  " + " ".join(cards_sorted))
    else:
        print("  (empty)")

# Help to let human player know what phase the game is in
def print_pass_phase_state(hearts_env, human_index: int, human_passed_cards: list[int]):
    print("\n=== PASS PHASE ===")
    print_full_hand(hearts_env, human_index)

    if human_passed_cards:
        print("\nCards you have selected to pass so far:")
        for idx in human_passed_cards:
            print(f"  {index_to_card_string(idx)} (index {idx})")
    else:
        print("\nYou have not selected any cards to pass yet.")       

# Help to let human player know what cards were passed to them
def print_pass_results(hearts_env, human_index: int):
    received = hearts_env.game.passingCards[human_index]  # list of Card objects
    if not received:
        print("\nYou did not receive any cards from passing this round (no-pass round or bug).")
        return 
    
    print("\n=== PASS RESULTS ===")
    print("You received the following cards from the pass:")
    for card in received:
        print(f"  {card}")  # Card.__str__() already prints like 'Qd', 'As'

# Helper to visualize to humans the current trick state
def print_current_trick_state(hearts_env):
    # Print the current trick number and the cards that have been played so far, in the order they were played.
    trick_num = hearts_env.game.trickNum
    trick_cards = hearts_env.game.currentTrick.trick

    # Collect only the played cards (non -1), in order
    played_cards = [c for c in trick_cards if c != -1]

    print(f"\n--- Trick {trick_num} ---")
    if not played_cards:
        print("No cards have been played in this trick yet.")
    else:
        print(f"Cards played so far ({len(played_cards)}/4):")
        for i, card_index in enumerate(played_cards, start=1):
            print(f"  {i}: {index_to_card_string(card_index)}")

# Define random policy for validation testing
def random_policy(obs: dict) -> int:
    # Initialize action mask
    action_mask = np.array(obs["action_mask"], dtype=np.int8)

    # Only need to mark the indices where cards are legal
    legal_indices = np.where(action_mask == 1)[0]

    # Debugging: If no legal actions, return none (shouldn't be possible)
    if len(legal_indices) == 0:
        return None
    
    # Otherwise, return random legal card:
    return int(np.random.choice(legal_indices))

# Define human policy to take human input
def human_policy(obs: dict) -> int:
    # Initialize action mask
    action_mask = np.array(obs["action_mask"], dtype=np.int8)

    # Ran into problems if the action mask was scalar, so have to convert if it is
    if action_mask.ndim == 0:
        print("Warning, action mask is scalar")
        action_mask = np.full(NUM_CARDS, action_mask, dtype=np.int8)

    # Only need to mark the indices where cards are legal - Implement for rest of hands?
    legal_indices = np.where(action_mask == 1)[0]

    # Debugging: If no legal actions, return none (Should not be possible)
    if len(legal_indices) == 0:
        print("No legal actions, returning None")
        return None

    # Print cards legal to play in current trick
    print("Legal cards:")
    for i, index in enumerate(legal_indices):
        print(f"    [{i}] {index_to_card_string(index)} (index {index})")

    # Have player choose a card by hand index number, making sure the choice is legal
    while True:
        try:
            choice_str = input(f"Select a card by number [0..{len(legal_indices) - 1}]: ")
            choice = int(choice_str)
            if 0 <= choice < len(legal_indices):
                chosen_action = int(legal_indices[choice])
                print(f"You played: {index_to_card_string(chosen_action)} (index {chosen_action})")
                return chosen_action
            else:
                print("Invalid choice number, try again: ")

        except ValueError:
            print("Please enter an interger")

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

# We have to discount rewards as earlier plays are more important than later plays
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
    global LEARNING_AGENT
    # Set Trajectories as variable that holds all important lists:
    trajectories = {"obs": [], "actions": [], "rewards": []}

    # Reset the environment for the new round
    hearts_env.reset()

    # We have to track rewards from last action taken by the learning player for that specific environment
    # Initialize that saved information here
    last_obs = None
    last_action = None

    # PettingZoo will loop over all agents in turn order until the hand is over
    for agent in hearts_env.agent_iter():
            # Get environmental data from agent whose turn it is
            obs, reward, terminated, truncated, info = hearts_env.last()

            # Print basic debug info
            # print(f"Step {step_count:3d} | Agent: {agent:8s} | "
            #     f"Reward: {reward:5.2f} | Terminated: {terminated} | Truncated: {truncated}")

            # We want to save the reward for the learning agent's previous action
            # This is applied later due to possibility of trick not having ended yet
            # If there is a previous action/environment:
            if agent == LEARNING_AGENT and last_obs is not None and last_action is not None:
                trajectories["obs"].append(last_obs)
                trajectories["actions"].append(last_action)
                trajectories["rewards"].append(float(reward))


            # If this agent is done for this hand then it takes no action
            if terminated or truncated:
                if agent == LEARNING_AGENT:
                    last_obs = None
                    last_action = None
                hearts_env.step(None)
                continue

            # Get observations
            obs_vector = obs["observation"]
            # Get action mask
            action_mask = obs["action_mask"]

            # Call the model and get the policy choice:
            action = modelPolicy(obs_vector, action_mask)

            # Save the learning agent's observations and action to the last turn variables
            if agent == LEARNING_AGENT:
                last_obs = obs_vector
                last_action = action
 

            # Send chosen action to env to take
            hearts_env.step(action)

    # Return the full set of trajectories for the current hand
    return trajectories

# Replaced REINFORCE with actor-critic method due to complexity of Hearts
# Used batches as lots of games were played and model wasn't learning as fast as hoped
def update_policy(batch_trajectories):

    # Initialize lists
    all_obs = []
    all_actions = []
    all_dis_rewards = []

    # Separate all the trajectory data from each batch
    for trajectory in batch_trajectories:
        # If learning agent took no actions due to early termination, skip update
        if len(trajectory["obs"]) == 0:
            print("No transitions for learning agent this round")
            continue

        # Initialize rewards array for each agent
        rewards = np.array(trajectory["rewards"], dtype=np.float32)

        # Call our discount function to modify the rewards
        discounted_rewards = discount_rewards(rewards, gamma=GAMMA)
        # Scale discounted_rewards to keep them in range of max points per hand
        discounted_rewards /= 26

        # Extend lists to hold new info
        all_obs.extend(trajectory["obs"])
        all_actions.extend(trajectory["actions"])
        all_dis_rewards.extend(list(discounted_rewards))

    # If no data in this batch, return
    if len(all_obs) == 0:
        return 0.0, 0.0, 0.0
    
    # Convert to arrays
    obs = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int32)
    discounted_rewards = np.array(all_dis_rewards, dtype=np.float32)

    # Convert obs and rewards to tensors for actor-critic
    dis_rewards_tf = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
    obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)

    # Calculate loss from chosen action
    with tf.GradientTape() as tape:
        # Forward pass: Policy_model re-outputs probabilities for all 52 cards from policy and value 
        probs = policy_model(obs_tf, training=True)
        values = tf.squeeze(value_model(obs_tf, training=True), axis=1)

        # Note probability of chosen action for policy
        action_one_hot = tf.one_hot(actions, NUM_CARDS)
        # Find probability of chosen actions at each time step
        chosen_action_probs = tf.reduce_sum(probs * action_one_hot, axis=1)
        # Convert the probabilities with log values for loss calculations
        log_probs = tf.math.log(chosen_action_probs + 1e-8)
            
        # Find advantage from discounted rewards - baseline of value model
        advantages = dis_rewards_tf - tf.stop_gradient(values)
        # Normalize advantages to reduce variance
        advantages = ((advantages - tf.reduce_mean(advantages)) / ((tf.math.reduce_std(advantages)) + 1e-8))

        # Find loss
        policy_loss = -tf.reduce_mean(log_probs * advantages)
        value_loss = tf.reduce_mean(tf.square(dis_rewards_tf - values))

        # Total loss
        loss = policy_loss + VALUE_LOSS_COEFF * value_loss

    # Now that we have computed loss, find and apply the gradients to both networks
    trainable_vars = policy_model.trainable_variables + value_model.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    # For batching, sum rewards for batch:
    total_reward = 0.0
    for trajectory in batch_trajectories:
        total_reward += float(np.sum(trajectory["rewards"]))

    total_dis_reward = float(np.sum(discounted_rewards))

    # Return stats in case we need them
    return float(loss.numpy()), total_reward, total_dis_reward

# The code for the model to validate against random players
def validation_against_random(num_hands):
    # Load the trained weights for the model
    load_trained_models()

    # Create a fresh environment
    hearts_env = env()

    print("Agents in this environment:", hearts_env.possible_agents)
    print(f"Trained agent seat (Validation): {VAL_AGENT}")
    print("Other three seats will play legal cards at random.\n")  

    num_players = 4
    # Track scores across hands as results of training
    cumulative_scores = np.zeros(num_players, dtype=np.int32) 

    # Play a number of hands equal to indicated value
    for hand in range(1, num_hands + 1):
        print(f"\n=== Evaluation Hand {hand} / {num_hands} ===")
        # Reset the environment to deal a new hand
        hearts_env.reset()

        # Play one full hand
        for agent in hearts_env.agent_iter():
            # Get obs from environment 
            obs, reward, terminated, truncated, info = hearts_env.last()

            # If the current agent is terminated/truncated, it cannot play
            # Shouldn't be possible, but just in case
            if terminated or truncated:
                hearts_env.step(None)
                continue

            # Choose action depending on which agent this is
            # If it's the trained agent:
            if agent == VAL_AGENT:
                # Trained policy
                obs_vec = obs["observation"]
                action_mask = obs["action_mask"]
                action = modelPolicy(obs_vec, action_mask)

            else:
                # Random opponent
                action = random_policy(obs)

            # Input card agent has selected to play
            hearts_env.step(action)

        # After the hand ends, read round and total scores
        round_scores = hearts_env.last_round_scores
        total_scores = [hearts_env.game.players[i].score for i in range(num_players)]

        # Update cumulative scores
        cumulative_scores += np.array(round_scores, dtype=np.int32)

        print(f"Round Scores: {round_scores}")
        print(f"Total Scores (engine): {total_scores}")

    # After all hands, compute average points per hand for each seat
    avg_points_per_hand = cumulative_scores / float(num_hands)

    # Print results
    print("\n=== Evaluation Summary vs Random Opponents ===")
    print(f"Hands played: {num_hands}")
    for i in range(num_players):
        agent_name = f"player_{i}"
        tag = ""
        if agent_name == VAL_AGENT:
            tag = " (TRAINED)"
        print(f"Seat {i}{tag}: total points = {cumulative_scores[i]}, "
              f"avg points/hand = {avg_points_per_hand[i]:.2f}")

    # Return values in case we need to save them
    return cumulative_scores, avg_points_per_hand     

# The code to play against the model as a human
def play_against_model(num_hands):
    # Create a fresh environment
    hearts_env = env()

    # Load trained weights
    load_trained_models()

    # Get the human's seat placement in the environment
    human_index = get_human_index(hearts_env)

    # For the total number of hands input to be played:
    for hand in range(1, num_hands + 1):
        print(f"\n=== Starting Hand {hand} ===")#
        # Reset environment for new hand
        hearts_env.reset()

        # Initialize list of passed cards and set pass_flag to false
        human_passed_cards: list[int] = []
        pass_results_shown = False

        # Loop over the hand until all agents are done
        for agent in hearts_env.agent_iter():
            # Get obs and reward info from the last step
            obs, reward, terminated, truncated, info = hearts_env.last()

            # If this agent is done, pass None
            if terminated or truncated:
                hearts_env.step(None)
                continue

            # If it's the human's turn
            if agent == HUMAN_AGENT:
                # Print game state
                phase = hearts_env.game.phase
                print_current_trick_state(hearts_env)

                # If it's the passing phase print the necessary info
                if phase == "Pass":
                    print_pass_phase_state(hearts_env, human_index, human_passed_cards)

                # If it's the play phase and the human hasn't seen the cards they were passed, show them
                else:
                    if not pass_results_shown:
                        print_pass_results(hearts_env, human_index)
                        pass_results_shown = True

                    # Otherwise show the current state of the game
                    print_current_trick_state(hearts_env)
                    print_full_hand(hearts_env, human_index)

                # Ask human for an action
                action = human_policy(obs)

                # If it's the pass phase, the action indicates a card to pass
                if phase == "Pass":
                    human_passed_cards.append(action)

            # If it's not the human's turn:
            else:
                # Use the trained model policy for all non-human agents
                obs_vec = obs["observation"]
                action_mask = obs["action_mask"]
                action = modelPolicy(obs_vec, action_mask)

            # Use the chosen action to take a step
            hearts_env.step(action)

        # After the hand ends, show the scores
        round_scores = [hearts_env.game.players[i].roundScore for i in range(4)]
        total_scores = [hearts_env.game.players[i].score for i in range(4)]

        print("\nHand complete.")
        print(f"Round Scores: {round_scores}")
        print(f"Total Scores: {total_scores}")



# Need to load trained model weights before running them
def load_trained_models():
    try:
        policy_model.load_weights(SAVE_PATH_POLICY)
        value_model.load_weights(SAVE_PATH_VALUE)
    except Exception as e:
        print("Could not load saved weights.")
        print("Error: ", e)
    
# The outside loop that runs the training
# Current number of episodes (rounds) set to 1000
def run_model(num_episodes):
    hearts_env = env()  # creates the wrapped PettingZoo env (with TerminateIllegalWrapper etc.)
    
    global LEARNING_AGENT
    # Set the agent that will learn here
    LEARNING_AGENT = hearts_env.possible_agents[0]

    # Variables to save scoring history and batch trajectories
    history_round_scores = []
    history_total_scores = []
    batch_trajectories = []
    batch_index = 0
    flag = True
    print_counter = 0

    # For each round that was set:
    for episode in range(1, num_episodes + 1):

        # Play hand and get trajectories, then batch them
        trajectories = play_hand(hearts_env)
        batch_trajectories.append(trajectories)

        # Get current scores from the HeartsEngine:
        round_scores = hearts_env.last_round_scores
        total_scores = [hearts_env.game.players[i].score for i in range(4)]

        # Save the current scores
        history_round_scores.append(round_scores)
        history_total_scores.append(total_scores)

        # Diagnostic log data, every 100 rounds, print the data for 10 rounds
        if episode % 100 == 0 or flag == True:     
            print(f"Episode {episode:4d}")
            print(f"Steps: {len(trajectories['obs']):3d}")
            print(f"Round Scores: {round_scores}")
            print(f"Total Score: {total_scores}")

            flag = True

            print_counter += 1
            if print_counter == 10:
                print_counter = 0
                flag = False

        # If the current batch size has been met, determine the round batch info
        if len(batch_trajectories) == BATCH_SIZE:
            batch_index += 1
            loss, batch_total_reward, batch_total_discounted_rewards = update_policy(batch_trajectories)
            print(f"Total Reward: {batch_total_reward:6.2f}")
            print(f"Loss: {loss:8.4f}")

            batch_trajectories = []

    # If the episodes end before the final batch can be filled, make a smaller batch
    if len(batch_trajectories) > 0:
        batch_index += 1
        loss, batch_total_reward, batch_total_discounted_rewards = update_policy(batch_trajectories)
        print(f"Total Reward: {batch_total_reward:6.2f}")
        print(f"Loss: {loss:8.4f}")
        
    # Save the scores:
    print("Saving weights from training")
    policy_model.save_weights(SAVE_PATH_POLICY)
    value_model.save_weights(SAVE_PATH_VALUE)

if __name__ == "__main__":
    # Ask whether the mode is training or playing against the human player
    mode = input("Enter 'train' to train the model, 'val' to validate the model's training, or 'play' to play against model: ").strip().lower()
    
    if mode == "train":
        run_model(num_episodes = 10000)

    elif mode == "val":
        validation_against_random(num_hands = 1000)

    elif mode == "play":
        play_against_model(num_hands = 1)

    else:
        print("Unknown mode")