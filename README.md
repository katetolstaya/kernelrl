# RLCore


## Dependencies
- Python
- OpenAI Gym
- SciPy

## Available algorithms

- Kernel Q-Learning with: 
    - Discrete states / actions
    - Continuous states / discrete actions
    - Continuous states and actions from [ACC 2018](https://arxiv.org/pdf/1804.07323.pdf)

- Kernel Normalized Advantage Functions continuous action spaces from [IROS 2018](https://katetolstaya.github.io/files/c_2018_tolstaya_etal_b.pdf)

- Also available: experience replay buffers

## To run

Kernel Q-Learning with Pendulum with prioritized experience replay
`python rlcore.py kq_pendulum_per.cfg`

Kernel NAF with Continuous Mountain Car
`python rlcore.py cfg/knaf_mcar.cfg`

Other options of configuration files are   
- Kernel Q-Learning for Cont. Mountain Car: `cfg/kq_mccar.cfg`
- Kernel Q-Learning for Pendulum: `cfg/kq_pendulum.cfg` 
- Kernel Q-Learning for discrete-action Cartpole: `cfg/kq_cartpole.cfg`
- Kernel NAF for Pendulum: `cfg/knaf_pendulum.cfg`
- `cfg/kqgreedy_pendulum_per.cfg`
- `cfg/kgreedyq_pendulum.cfg`


## Tuning parameters
To tune learning rates and other parameters, adjust the corresponding parameters in the .cfg file.

