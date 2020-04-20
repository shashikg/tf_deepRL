import tf_easyRL

class Env(object):
    observation_space = None
    action_space = None
    provider = None
    name = None

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode="display"):
        raise NotImplementedError

    def seed(self, seed=None):
        raise NotImplementedError

    def close(self):
        pass
