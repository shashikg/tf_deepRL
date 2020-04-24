import tf_deepRL
import tensorflow as tf
from skvideo.io import FFmpegWriter as VideoWriter
import time, io, base64
from IPython import display as ipy_display
import numpy as np
from tqdm.notebook import tqdm as Ntqdm
from tf_deepRL.utils import DisplayTrainStatus
import time
from PIL import Image
import PIL
import matplotlib.pyplot as plt

supported_status_type = ['text', 'plot']

class Memory:
  def __init__(self):
      self.clear()

  def clear(self):
      self.states = []
      self.rewards = []
      self.actions = []

  def add(self, new_state, new_reward, new_action):
      self.states.append(new_state)
      self.rewards.append(new_reward)
      self.actions.append(new_action)

class Agent(object):
    model = None
    memory = None
    gamma = None
    env = None

    def discount_rewards(self):
        discounted_rewards = np.zeros_like(self.memory.rewards)
        R = 0
        for t in range(len(self.memory.rewards)-1, -1, -1):
            R = self.memory.rewards[t] + R*self.gamma
            discounted_rewards[t] = R

        return discounted_rewards

    def take_action(self, state):
        raise NotImplementedError

    def calc_loss(self):
        raise NotImplementedError

    def train(self, opt, episodes, status_type="text", status_interval = 1):
        if status_type in supported_status_type:
            pass
        else:
            err_txt = ""
            for temp in supported_status_type: err_txt += "'" + temp + "'" + " or "
            raise KeyError("Invalid status type. " + "Use following status type: " + err_txt[:-4] + ".")

        history = {}
        history['total_reward'] = []
        dts = DisplayTrainStatus(episodes, status_interval, status_type)

        for e in range(episodes):
            state = self.env.reset()
            self.memory.clear()

            while True:
                new_state, reward, action, done = self.take_action(state)
                self.memory.add(state, reward, action)

                if done:
                    total_reward = sum(self.memory.rewards)
                    history['total_reward'].append(total_reward)

                    with tf.GradientTape() as tape:
                        loss = self.calc_loss()

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    opt.apply_gradients(zip(grads, self.model.trainable_variables))
                    dts.display(history)

                    break

                state = new_state

        return history

    def play(self, max_steps, stop_when_finish=False):
        self.env.start_pyglet_display()

        state = self.env.reset()
        for step in range(max_steps):
            rendered_state, mode = self.env.render()
            new_state, reward, action, done = self.take_action(state)
            state = new_state

            if mode == "img":
                ipy_display.clear_output(wait=True)
                im = Image.fromarray(rendered_state)
                ipy_display.display(im)
            elif mode == "io":
                ipy_display.clear_output(wait=True)
                print(rendered_state)

            if stop_when_finish and done:
                if self.env.virtual_display == False:
                    time.sleep(2)
                break

        self.env.close()

        return True
