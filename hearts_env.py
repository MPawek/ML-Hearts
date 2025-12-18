# noqa: D212, D415
# My modification of the PettingZoo gin_rummy code with Hearts code from https://github.com/danielcorin/Hearts

from __future__ import annotations

import gymnasium
import numpy as np
import HeartsEngine as HE
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

from Card import Card, Suit, Rank

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-10)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "hearts_v1",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self,
        render_mode: str | None = None
    ):
        EzPickle.__init__(
            self,
            render_mode=render_mode
        )

        self.render_mode = render_mode

        # Global list of all agents that could be used (4 for the 4 players)
        # agents are the agents active in this episode (will always be all 4)
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agents = self.possible_agents[:]
    
    # we’ll map cards to indices 0–51
        self.num_cards = 52
        # Cards in hand, cards played, round scores (normalized?), trick suit, heartsBroken, pass/play
        # Obs must keep track of:
        # Pass/Play - Boolean
        # heartsBroken - Boolean
        # Trick suit - Binary indicator in 1 of 4 places
        # Cards in Hand - Binary size 52 array
        # Cards in trick - Binary size 52 array
        # Round scores (normalized) - Normalized to be between 0 and 1, one spot for each player
        obs_len = 1 + 1 + 4 + 1 + self.num_cards + self.num_cards + 4
        # May want to include previously played cards in other tricks later 

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=1,
                        shape=(obs_len,),
                        dtype=np.int8,
                    ),
                    "action_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.num_cards,),
                        dtype=np.int8,
                    ),
                }
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(self.num_cards) for agent in self.possible_agents
        }

        # Cycles through agents during trick in order
        self._agent_selector = agent_selector(self.agents)
        # Determines whose turn it currently is
        self.agent_selection = self._agent_selector.next()

        self.rewards = {a: 0.0 for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}

        # This will be the Hearts game engine instance
        self.game = HE.Hearts()
        self.screen = None
        self.save_states = None  # optional, if add pygame rendering later
        self.switch_flag = False

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # Turns agent string into player index
    def agentPlayerIndex(self, agent: str) -> int:
        return self.possible_agents.index(agent)
    
    # Turns player index into agent string
    def playerAgentIndex(self, index: int) -> str:
        return self.possible_agents[index]


    # Resets environment back to base environment - starting hand from scratch
    def reset(self, seed=None, options=None):

        # All possible agents are the current agents
        self.agents = self.possible_agents[:]

        # Rewards start at zero during new environment
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

        # No agent is done or truncated at start 
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}

        # Determine environment info for each agent
        self.infos = {a: {} for a in self.possible_agents}

        # Set up hands for next round
        self.game.newRound()
        self.switch_flag = False

        # Determine agent order and determine who is currently up.
        self._agent_selector = agent_selector(self.agents) 
        self.agent_selection = self._agent_selector.next()

        
    def step(self, action):
        # Figure out which agent goes next
        agent = self.agent_selection
        player_index = self.agentPlayerIndex(agent)

        # if this agent is already done, follow PettingZoo dead-step pattern
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Clear previous rewards by giving everyone zero
        for player in self.possible_agents:
            self.rewards[player] = 0

        # Apply the action to Hearts game
        # First, check whether we're in Pass or Play
        # If passing:
        if self.game.phase == "Pass":
            # Choose cards to pass and record
            # Agents pick one at a time, so we have to input their choices into the list
            self.game.playersPassCards(player_index, action)

            # If play phase starts, change next agent to 2 of clubs holder
            if self.game.phase == "Play":
                # Get starting player's index
                winner_index = self.game.trickWinner
                # Get starting player's name
                winner_agent = self.playerAgentIndex(winner_index)

                # Find starting player's place in current turn order
                position = self.agents.index(winner_agent)

                # For debugging, prints who won and the order of players
                # print(f"Trick winner index: {winner_index}, winner_agent: {winner_agent}")
                # print("Order before:", self.agents)

                # Reorder agents so starting player is first
                new_order = self.agents[position:] + self.agents[:position]

                # Keep agents in sync with order selector
                self.agents = new_order
                self._agent_selector.reinit(new_order)

                # Reset turn order to start with first player
                self.agent_selection = self._agent_selector.reset()

                # For debugging, prints new order of players
                # print("Order after :", new_order)

                # Set switch_flag to true so we don't iterate player order right after we changed it
                self.switch_flag = True

        # If playing:
        else:
            # Play the given card
            self.game.playTrick(player_index, action)

            # If the last card in the trick was just played, set agent to who will start next round.
            # The only case that a trick has zero cards after a play is if the last card was played
            if self.game.currentTrick.cardsInTrick == 0:

                # Get winning player's index
                winner_index = self.game.trickWinner
                winner_agent = self.playerAgentIndex(winner_index)

                # Get points in trick and scale them
                points = self.game.lastTrickPoints / -26

                # Reset rewards for all players
                for agent in self.possible_agents:
                    self.rewards[agent] = 0.0

                # Assign reward points here
                self.rewards[winner_agent] += points

                # Simple Reward Shaping: Assign some points if agent took hearts or Queen this step:
                trick_cards = self.game.currentTrick.trick
                num_hearts = sum(1 for card in trick_cards if card // 13 == 3)
                has_queen = any(card == 36 for card in trick_cards)

                # Small penalty for heart taken
                self.rewards[winner_agent] += -0.1 * num_hearts

                if has_queen:
                    self.rewards[winner_agent] += -0.5

                # Debugging:
                # print(f"Trick winner index: {winner_index}, winner_agent: {winner_agent}")
                # print("Order before:", self.agents)

                # Reorder agents so winning player is first
                position = self.agents.index(winner_agent)
                new_order = self.agents[position:] + self.agents[:position]
                self.agents = new_order
                self._agent_selector.reinit(new_order)

                # Reset turn order to start with winning player
                self.agent_selection = self._agent_selector.reset()

                # Debugging:
                # print("Order after :", new_order)

                # Set switch_flag to true so we don't iterate player order right after we changed it
                self.switch_flag = True

                
            # Determine if hand is over (trickNum == 13)
            if (self.game.trickNum == 13 and self.game.currentTrick.cardsInTrick == 0):

                # Set termination[agent] = True for all agents
                for player in self.possible_agents:
                    self.terminations[player] = True

                # Reset agents list
                self.agents = []

                # If hand is over, finalize points (e.g. shooting the moon)
                # If anyone shot the moon:
                if any(aPlayer.roundScore == 26 for aPlayer in self.game.players):
                    # For each player:
                    for player in self.possible_agents:
                        # If that player got 26 points, erase their negative rewards and give them a positive one scaled the same as the other points
                        if self.game.players[self.agentPlayerIndex(player)].roundScore == 26:
                            self.rewards[player] += 39/26

                        # Otherwise, all other players get equivalent negative reward
                        else:
                            self.rewards[player] -= 1
                
                # Save round scores
                round_scores = [self.game.players[i].roundScore for i in range(4)]

                # Use engine to handle end-of-hand scoring
                self.game.handleScoring()

                # Save scores 
                self.last_round_scores = round_scores

        # Accumulate rewards (PettingZoo machinery)
        self._accumulate_rewards()

        # Advance to next agent
        if len(self.agents) > 0 and not self.switch_flag:
            self.agent_selection = self._agent_selector.next()

        # Set switch_flag to false to reset agent selector
        self.switch_flag = False


    def observe(self, agent):
        # ---- Build obs from the Hearts game state ----
        # Cards in hand, cards played, round scores (normalized?), trick suit, heartsBroken, pass/play
        # Obs must keep track of:
        # Pass/Play - Boolean
        # heartsBroken - Boolean
        # first Trick - Boolean
        # Trick suit - Binary indicator in 1 of 4 places
        # Cards in Hand - Binary size 52 array
        # Cards in trick - Binary size 52 array
        # Round scores (normalized) - Normalized to be between 0 and 1, one spot for each player
        obs = np.zeros(1 + 1 + 4 + 1 + self.num_cards + self.num_cards + 4, dtype=np.float32)

        # If any players have rterminated, they have no legal actions
        if self.terminations[agent] or self.truncations[agent]:
            action_mask = np.zeros(self.num_cards, dtype=np.float32)
            return {"observation": obs, "action_mask": action_mask}
        
        # Get current player and that player's legal moves
        cur_agent = self.agentPlayerIndex(agent)
        action_mask = self.game.isLegal(cur_agent)

        # For debugging: if no legal moves (usually impossible), print all diagnostic data
        if all(option == 0 for option in action_mask):
            print('Error: No legal moves\n')
            print(f'Agent: {agent}\n')
            print(f'Phase: {self.game.phase}\n')
            print(f'Trick Number: {self.game.trickNum}\n')
            print(f'Cards in the current trick: {self.game.currentTrick.cardsInTrick}')
            for card in range(4):
                print(f'{self.game.currentTrick.trick[card]}\n')
            print(f'Current Suit: {self.game.currentTrick.suit}\n')
            print(f'Are hearts broken: {self.game.heartsBroken}\n')
            print('Cards in hand: [')
            for index in range(len(self.game.players[cur_agent].hand.hand)):
                print(f'{self.game.players[cur_agent].hand.hand[index]}, ')

        # Mark if game is currently in play or pass phase
        if self.game.phase == "Play":
            obs[0] = 1

        # Mark if hearts have been broken
        if self.game.heartsBroken:
            obs[1] = 1

#   Implemented in one line below
#        if self.game.currentTrick.suit == Suit(HE.clubs):
#            obs[3] = 1
#        elif self.game.currentTrick.suit == Suit(HE.diamonds):
#            obs[4] = 1
#        elif self.game.currentTrick.suit == Suit(HE.spades):
#            obs[5] = 1
#        elif self.game.currentTrick.suit == Suit(HE.hearts):
#            obs[6] = 1
#        else:
#            obs[2] = 1

        # Determine current trick suit (or if no one has played a card yet)
        obs[3 + self.game.currentTrick.suit] = 1

        # If a player has a card in hand, mark it in the obs list
        for aCard in range(len(self.game.players[cur_agent].hand.hand)):
            if self.game.players[cur_agent].hand.hand[aCard] == 1:
                obs[7 + aCard] = 1

        # Mark all cards currently in trick
        for aCard in self.game.currentTrick.trick:
            if aCard >= 0:
                obs[7 + self.num_cards + aCard] = 1

        # Note each player's round score, normalized to be between 0 and 1
        for i in range(4):
            obs[7 + self.num_cards + self.num_cards + i] = self.game.players[i].roundScore / 26

        # Return environment and action mask
        return {"observation": obs, "action_mask": action_mask}


    # Rendering is not implemented, game is currently text-based
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
'''
        def draw_borders(x, y, width, height, bw, color):
            pygame.draw.line(
                self.screen, color, (x - bw // 2 + 1, y), (x + width + bw // 2, y), bw
            )
            pygame.draw.line(
                self.screen,
                color,
                (x - bw // 2 + 1, y + height),
                (x + width + bw // 2, y + height),
                bw,
            )
            pygame.draw.line(
                self.screen, color, (x, y - bw // 2 + 1), (x, y + height + bw // 2), bw
            )
            pygame.draw.line(
                self.screen,
                color,
                (x + width, y - bw // 2 + 1),
                (x + width, y + height + bw // 2),
                bw,
            )

        screen_height = self.screen_height
        screen_width = int(screen_height * (1 / 20) + 3.5 * (screen_height * 1 / 2))

        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Gin Rummy")
            else:
                self.screen = pygame.Surface((screen_width, screen_height))

        # Setup dimensions for card size and setup for colors
        tile_size = screen_height * 2 / 10

        bg_color = (7, 99, 36)
        white = (255, 255, 255)
        self.screen.fill(bg_color)

        # Load and blit all images for each card in each player's hand
        for i, player in enumerate(self.possible_agents):
            state = self.env.game.get_state(self._name_to_int(player))

            # This is to mitigate the issue of a blank board render without cards as env returns an empty state on
            # agent termination. Used to store states for renders when agents are terminated
            if len(state) == 0:
                state = self.save_states
            else:
                self.save_states = state

            for j, card in enumerate(state["hand"]):
                # Load specified card
                card_img = get_image(os.path.join("img", card + ".png"))
                card_img = pygame.transform.scale(
                    card_img, (int(tile_size * (142 / 197)), int(tile_size))
                )
                # Players with even id go above public cards
                if i % 2 == 0:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                calculate_width(
                                    self.possible_agents,
                                    screen_width,
                                    i,
                                    tile_size,
                                    tile_scale=31,
                                )
                                - calculate_offset(state["hand"], j, tile_size)
                            ),
                            calculate_height(screen_height, 4, 1, tile_size, -1),
                        ),
                    )
                # Players with odd id go below public cards
                else:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                calculate_width(
                                    self.possible_agents,
                                    screen_width,
                                    i,
                                    tile_size,
                                    tile_scale=31,
                                )
                                - calculate_offset(state["hand"], j, tile_size)
                            ),
                            calculate_height(screen_height, 4, 3, tile_size, 0),
                        ),
                    )

            # Load and blit text for player name
            font = get_font(os.path.join("font", "Minecraft.ttf"), 36)
            text = font.render("Player " + str(i + 1), True, white)
            textRect = text.get_rect()
            if i % 2 == 0:
                textRect.center = (
                    (
                        screen_width
                        / (np.ceil(len(self.possible_agents) / 2) + 1)
                        * np.ceil((i + 1) / 2)
                    ),
                    calculate_height(screen_height, 4, 1, tile_size, -(22 / 20)),
                )

            else:
                textRect.center = (
                    (
                        screen_width
                        / (np.ceil(len(self.possible_agents) / 2) + 1)
                        * np.ceil((i + 1) / 2)
                    ),
                    calculate_height(screen_height, 4, 3, tile_size, (23 / 20)),
                )
            self.screen.blit(text, textRect)

            for j, card in enumerate(state["top_discard"]):
                card_img = get_image(os.path.join("img", card + ".png"))
                card_img = pygame.transform.scale(
                    card_img, (int(tile_size * (142 / 197)), int(tile_size))
                )

                self.screen.blit(
                    card_img,
                    (
                        (
                            (
                                ((screen_width / 2) + (tile_size * 31 / 616))
                                - calculate_offset(state["top_discard"], j, tile_size)
                            ),
                            calculate_height(screen_height, 2, 1, tile_size, -(1 / 2)),
                        )
                    ),
                )

            # Load and blit discarded cards
            font = get_font(os.path.join("font", "Minecraft.ttf"), 36)
            text = font.render("Top Discarded Card", True, white)
            textRect = text.get_rect()
            textRect.center = (
                (
                    calculate_width(
                        self.possible_agents, screen_width, 0, tile_size, tile_scale=31
                    )
                ),
                calculate_height(screen_height, 2, 1, tile_size, (-2 / 3))
                + (tile_size * (13 / 200)),
            )
            self.screen.blit(text, textRect)

            draw_borders(
                x=int((screen_width / 2) + (tile_size * 31 / 616))
                - int(tile_size * 23 / 56)
                - 5,
                y=calculate_height(screen_height, 2, 1, tile_size, -(1 / 2)) - 6,
                width=int(tile_size) - int(tile_size * (9 / 40)),
                height=int(tile_size) + 10,
                bw=3,
                color="white",
            )

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
    
    def close(self):
        self.screen = None 
        '''