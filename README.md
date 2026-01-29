This is the first stage of my project to make an ML model that uses reinforcement learning to play the card game Hearts. This stage uses an actor-critic model to train against 3 opponents playing cards at random. There is also the option for a human to play, inputting their card choices into the terminal. The next stage will likely be to train the agent against older versions of itself to give it more of a challenge. In the future, implementing some form of memory will likely be my objective so the model can keep track of which cards have already been played, and use the potential cards remaining to inform it's decisions.

Implemented using TensorFlow.

Hearts implementation borrowed from https://github.com/danielcorin/Hearts
