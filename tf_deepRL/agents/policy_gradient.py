import tensorflow as tf
import numpy as np
from tf_deepRL.agents.core import Agent, Memory

class VANILLA(Agent):
    def __init__(self, policy_model, env, gamma=0.95):
        self.model = policy_model
        self.memory = Memory()
        self.gamma = gamma
        self.env = env

    def take_action(self, state):
        logits = self.model.predict(state)
        policy = tf.nn.softmax(logits).numpy()
        action = np.random.choice(self.env.action_space, p=policy.flatten())

        new_state, reward, done = self.env.step(action)

        return new_state, reward, action, done

    def calc_loss(self):
        state_t = np.vstack(self.memory.states)
        discounted_reward_t = self.discount_rewards()
        action_t = np.array(self.memory.actions)

        logits = self.model(state_t)
        neg_log = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_t)
        loss = tf.reduce_mean(neg_log*discounted_reward_t)

        return loss
