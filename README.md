# kernelrl
Reinforcement learning using kernel-based function approximation

To run experiments, do:
`python3 rlcore.py cfg/exp/p1.cfg`

And use the testing loss and reward mean and standard deviation values.


Change `continuous_mountain_car.py`:

``self.state = np.array([self.np_random.uniform(low=-1.2, high=0.45), 0])``
