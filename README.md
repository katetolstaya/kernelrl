# rlcore




# Dependencies
- Python
- OpenAI Gym
- SciPy

# Available Algorithms

- Kernel Q-Learning with: 
    - Discrete State/Actions
    - Cont. State /Discrete Actions
    - Cont. State and Actions

- Kernel Normalized Advantage Functions for Continuous State/Action Spaces

- Also available: experience replay buffers

# To run:

Kernel Q-Learning with Pendulum and replay buffer
`python rlcore.py kq_pendulum_per.cfg`

Kernel NAF with Continuous Mountain Car
`python rlcore.py cfg/knaf_mcar.cfg`


# Tuning parameters:
To tune learning rates and other parameters, adjust the corresponding parameters in the .cfg file.

# TODO
- What is the difference between kqgreedy_replay and kqlearning_replay?