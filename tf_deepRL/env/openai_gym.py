import tf_deepRL
from tf_deepRL.env.core import Env
import gym
from pyvirtualdisplay import Display
import numpy as np
import pyglet

supported_env = ['CartPole-v0']

class gymEnv(Env):
    def __init__(self, name=None, virtual_display=False):
        if name in supported_env:
            pass
        else:
            err_txt = ""
            for temp in supported_env: err_txt += temp + ", "
            raise KeyError("This environment is not supported | " + "Right now only following environments are supported: " + err_txt[:-2])

        self.env = gym.make(name)

        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space.n
        self.provider = "OpenAI-gym"
        self.name = name
        self.virtual_display_flag = virtual_display
        self.virtual_display = None
        self.pyglet_display = None

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

    def render(self):
        if self.pyglet_display is None:
            self.start_pyglet_display()

        if self.virtual_display_flag:
            modes = self.env.metadata['render.modes']
            if "ansi" in modes:
                rendered_state = self.env.render(mode="ansi")
                mode = "io"
            elif "rgb_array" in modes:
                rendered_state =  self.env.render(mode="rgb_array")
                mode = "img"

            return rendered_state, mode
        else:
            return self.env.render(mode="human"), "display"

    def seed(self, seed=None):
        return self.env.seed(seed)

    def close(self):
        if self.env.viewer:
            pyglet.canvas._displays.remove(self.env.viewer.window.display)
        self.env.close()

        if self.virtual_display_flag and self.virtual_display != None:
            self.virtual_display.stop()
            self.virtual_display = None

        self.pyglet_display = None

    def start_pyglet_display(self):
        self.pyglet_display = {}
        if self.virtual_display_flag:
            self.virtual_display = Display(visible=0, size=(400, 300))
            self.virtual_display.start()
            self.pyglet_display['display'] = pyglet.canvas.Display(name=self.virtual_display.cmd_param[-1])
        else:
            self.pyglet_display['display'] = pyglet.canvas.Display(name=":0")

        self.pyglet_display['screen'] = self.pyglet_display['display'].get_screens()
        self.pyglet_display['config'] = self.pyglet_display['screen'][0].get_best_config()
        self.pyglet_display['context'] = self.pyglet_display['config'].create_context(None)

        pyglet.gl.current_context = self.pyglet_display['context']
