import numpy as np
from policy_evaluation import PolicyEvaluationBase
from util import configure_name, ScheduledParameter
from corerl.kernel import make_kernel
import json

# ------------------------------
# RBF-grid GTD Policy Evaluation
# ------------------------------

@configure_name('RBFGTD')
class RBFGTDPolicyEvaluation(PolicyEvaluationBase):
    def __init__(self, stateCount, config):
        self.kernel = make_kernel(config)
        # Bounds on the observation space
        bounds     = json.loads(config.get('ObservationBounds'))
        assert(len(bounds) == stateCount)
        # How much we want to shrink the spacing between grid points
        scale      = config.getfloat('GridScaleFactor', 1.0)
        # ---- Build our grid of RBF points ----
        if len(self.kernel.sig.shape) == 0:
            sig = self.kernel.sig * np.ones(stateCount)
        else:
            sig = self.kernel.sig
        axes  = [np.arange(bounds[i][0], bounds[i][1], scale*sig[i]) for i in range(stateCount)]
        grids = np.meshgrid(*axes)
        # Our basis point grid
        self.D = np.stack([g.ravel() for g in grids], axis=1)
        # ----
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
        # TD-loss expectation approximation rate
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Learning rate
        self.eta  = ScheduledParameter('LearningRate', config)
        # Running estimate of our expected TD-loss and parameters
        self.y  = np.zeros(len(self.D))
        self.th = np.zeros(len(self.D))
    def bellman_error(self, s, a, r, s_):
        if s_ is None:
            phi = self.kernel.f(s, self.D).flatten()
            return r - self.th.dot(phi)
        else:
            phi  = self.kernel.f(s,  self.D).flatten()
            phi_ = self.kernel.f(s_, self.D).flatten()
            return r + self.gamma * self.th.dot(phi_) - self.th.dot(phi)
    def model_error(self):
        return 0.
    def train(self, step, sample):
        self.eta.step(step)
        self.beta.step(step)
        # Unpack sample
        s, a, r, s_ = sample
        if s_ is None:
            phi = self.kernel.f(s, self.D).flatten()
            delta = r - self.th.dot(phi)
            self.y  += self.beta.value * (delta * phi - self.y)
            self.th += self.eta.value * phi.dot(self.y) * phi
            delta_ = r - self.th.dot(phi)
        else:
            phi  = self.kernel.f(s,  self.D).flatten()
            phi_ = self.kernel.f(s_, self.D).flatten()
            delta = r + self.gamma * self.th.dot(phi_) - self.th.dot(phi)
            self.y  += self.beta.value * (delta * phi - self.y)
            self.th += self.eta.value * phi.dot(self.y) * (phi - self.gamma * phi_)
            delta_ = r + self.gamma * self.th.dot(phi_) - self.th.dot(phi)
        # Compute new error
        loss = 0.5*delta_**2
        return (float(loss), float(len(self.D)))
    def test(self, s, v):
        return np.mean(np.abs(self.kernel.f(np.array(s), self.D).dot(self.th).flatten() - v) / np.abs(v))
    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')
