import tf_deepRL

class Env(object):
    state_space = None
    action_space = None
    provider = None
    name = None
    env = None

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode="display"):
        raise NotImplementedError

    def seed(self, seed=None):
        return

    def close(self):
        pass

    def ret_orig_env(self):
        return self.env
