import tf_deepRL
import tensorflow as tf
from skvideo.io import FFmpegWriter as VideoWriter
import time, io, base64
from IPython import display

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
            R = self.rewards[t] + R*self.gamma
            discounted_rewards[t] = R

        return discounted_rewards

    def take_action(self, state):
        raise NotImplementedError

    def calc_loss(self):
        raise NotImplementedError

    def train(self, opt, episodes):
        history = {}
        history['loss'] = []
        history['total_reward'] = []

        for e in range(episodes):
            state = self.env.reset()
            self.memory.clear()

            while True:
                new_state, reward, action, done = self.take_action(state)
                self.memory.add(new_state, reward, action)

                if done:
                    total_reward = sum(self.memory.rewards)
                    history['total_reward'].append(total_reward)

                    with tf.GradientTape() as tape:
                        loss = self.calc_loss()
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        opt.apply_gradients(zip(grads, self.model.trainable_variables))

                        history['loss'].append(loss.numpy())

                    print("| Episode:", e, " | Loss:", history['loss'][-1], " | Total Rewards:", history['total_reward'][-1], "|")

                    break

                state = new_state

        return history

    def play(self, max_steps, stop_when_finish=False, IPython_flag=False):
        rendered_states = []
        state = self.env.reset()
        for step in range(max_steps):
            rendered_state, mode = self.env.render(IPython_flag)
            rendered_states.append(rendered_state)
            new_state, reward, action, done = self.take_action(state)

            if stop_when_finish and done:
                if !IPython_flag:
                    time.sleep(2)
                self.env.close()
                break

        if mode == "io":
            for rendered_state in rendered_states:
                display.clear_output(wait=True)
                print(rendered_state)
                time.sleep(1/5.0)
        elif mode == "img":
            filename = self.env.name + ".mp4"
            video = VideoWriter(filename)
            for rendered_state in rendered_states:
                video.writeFrame(rendered_state)
            video.close()

            return filename

        return True
