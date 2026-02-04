# ML-Hearts: Reinforcement Learning Card Game Agent

## Overview
ML-Hearts is a **reinforcement learning project** that trains an agent to play a trick-taking card game environment. The focus of the project is on **practical RL experimentation**, including environment interaction, reward design, training stability, and measurable performance evaluation.

This project was completed as an **open-ended final project**, where the high-level scope was provided, but the specific approach, model design, and evaluation criteria were defined independently.

---

## Project Goals
- Design and train a reinforcement learning agent for a non-trivial game environment
- Explore reward shaping and training stability
- Evaluate agent performance using measurable outcomes
- Gain hands-on experience with iterative ML experimentation

---

## Tech Stack
- **Language:** Python  
- **Libraries:** Gym-style environment, reinforcement learning frameworks  
- **Training:** Policy-based / value-based RL (implementation-specific)  
- **Evaluation:** Win-rate and performance metrics over repeated games  

---

## Environment & Problem Setup

### Game Environment
The agent interacts with a card game environment adapted from an existing Gym-compatible implementation. Observations encode the current game state, while actions correspond to valid moves available to the agent at each turn.

### State and Action Space
- **State:** Encoded representation of the agent’s hand, played cards, and game context
- **Actions:** Discrete set of legal card plays
- Invalid actions are filtered to ensure rule-compliant gameplay

---

## Model & Training Approach

### Learning Strategy
The agent was trained using reinforcement learning techniques that balance **exploration and exploitation**. Multiple training runs were performed to tune hyperparameters and improve convergence behavior.

### Reward Design
The reward function was iteratively refined to:
- encourage legal and strategically sound plays,
- penalize unfavorable outcomes,
- promote long-term performance rather than short-term gains.

This process was critical to achieving stable learning in a stochastic environment.

---

## Training & Evaluation

### Training Process
- Multiple training episodes were executed against baseline opponents
- Model checkpoints were evaluated periodically
- Performance trends were tracked across training iterations

### Evaluation Metrics
Agent performance was evaluated using:
- average score per game,
- win-rate against baseline players,
- consistency across repeated trials.

These metrics provided quantitative insight into learning progress and policy quality.

---

## Results
The trained agent demonstrated **measurable improvement over baseline behavior**, achieving stronger average outcomes as training progressed. While performance varied due to the stochastic nature of the environment, results showed that the agent learned effective strategies rather than random play.

---

## Code Attribution & Academic Context
This project builds upon an existing Faruma Gym card game environment, which is cited in the header of relevant code. Portions of the environment and supporting utilities were adapted to support training and evaluation.

All model design choices, training logic, experimentation, and analysis were implemented independently.

---

## What This Project Demonstrates
- Practical application of **reinforcement learning**
- Experience working with **game environments and action constraints**
- Iterative experimentation with reward functions and hyperparameters
- Ability to evaluate ML systems using quantitative metrics
- Comfort navigating non-deterministic training behavior

---

## Limitations & Future Work
- Train against stronger or adaptive opponents
- Improve state representation
- Explore alternative RL algorithms
- Extend evaluation to longer training horizons
- Implement a rudimentary memory allowing the agent to recall cards previously played

---

## Author
Developed independently by **Montana Pawek**.

