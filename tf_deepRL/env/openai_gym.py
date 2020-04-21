import tf_deepRL
from tf_deepRL.env.core import Env
import gym
from pyvirtualdisplay import Display
import numpy as np

supported_env = ['CartPole-v0']

class gymEnv(Env):
    def __init__(self, name=None):
        if name in supported_env:
            pass
        else:
            err_txt = ""
            for temp in supported_env: err_txt += temp + ", "
            raise KeyError("This environment is not supported | " + "Supported environment: " + err_txt[:-2])

        self.env = gym.make(name)

        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space.n
        self.provider = "OpenAI-gym"
        self.name = name

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if self.name == "CartPole-v0":
            state = np.expand_dims(state, axis=0)

        return state, reward, done

    def reset(self):
        state = self.env.reset()

        if self.name == "CartPole-v0":
            state = np.expand_dims(state, axis=0)

        return state

    def render(self, IPython_flag=False):
        if IPython_flag:
            display = Display(visible=0, size=(400, 300))
            display.start()

            modes = self.env.metadata['render.modes']

            if "ansi" in modes:
                rendered_state = self.env.render(mode="ansi")
                mode = "io"
            elif "rgb_array" in modes:
                rendered_state =  self.env.render(mode="rgb_array")
                mode = "img"

            display.stop()
            return rendered_state, mode
        else:
            return self.env.render(mode="human"), "display"

    def seed(self, seed=None):
        return self.env.seed(seed)

    def close(self):
        self.env.close()
