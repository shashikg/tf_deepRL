from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

class DisplayTrainStatus:
    def __init__(self, max_eps, status_interval, type):
        self.txt_till_now = []
        self.max_eps = max_eps
        self.status_interval = status_interval
        self.type = type
        self.txt_l = "                           "
        self.txt_r = "==========================="
        self.txt_len = len(self.txt_l)
        self.last_time = time.time()

    def text(self, history):
        row = int(len(history['total_reward'])/self.status_interval)
        col = len(history['total_reward'])%self.status_interval
        for rows in self.txt_till_now:
            print(rows)

        if col == 0:
            sec = int((time.time()-self.last_time)/self.status_interval)
            ms = int((time.time()-self.last_time - sec*self.status_interval)*1000/self.status_interval)
            print("Episode:", str(len(history['total_reward'])) + "/" + str(self.max_eps))
            print("["+ self.txt_r +"] - " + str(sec) + "s " + str(ms) + "ms/episode - rewards:", history['total_reward'][-1])
            self.txt_till_now.append("Episode: " + str(len(history['total_reward'])) + "/" + str(self.max_eps))
            self.txt_till_now.append("[" + self.txt_r + "] - " + str(sec) + "s " + str(ms) + "ms/episode - rewards: " + str(np.mean(history['total_reward'][-self.status_interval:])))
            self.last_time = time.time()
        else:
            sec = int((time.time()-self.last_time)/col)
            ms = int((time.time()-self.last_time - sec*col)*1000/col)
            i = int(col*self.txt_len/self.status_interval)
            print("Episode:", str(len(history['total_reward'])) + "/" + str(self.max_eps))
            print("["+ self.txt_r[:i] + self.txt_l[i:] +"] - " + str(sec) + "s " + str(ms) + "ms/episode - rewards:", history['total_reward'][-1])

    def plot(self, history):
        plt.cla()
        x = np.array(history['total_reward'])

        sec = int((time.time()-self.last_time)/len(x))
        ms = int((time.time()-self.last_time - sec*len(x))*1000/len(x))

        t = time.time()

        if len(x)>self.status_interval+1:
            for i in range(self.status_interval):
                x[self.status_interval-1:] += np.array(history['total_reward'][self.status_interval-1-i:len(x)-i])

        x[self.status_interval-1:] = x[self.status_interval-1:]/self.status_interval

        plt.plot(x, color="black", linewidth=1)
        plt.xlabel("Episode")
        plt.ylabel("Total Rewards")
        plt.title(str(sec) + "s " + str(ms) + "ms/episode")
        display.display(plt.gcf())

        # self.last_time = time.time()

    def display(self, history):
        display.clear_output(wait=True)
        if self.type == "text":
            self.text(history)
        elif self.type == "plot":
            self.plot(history)
