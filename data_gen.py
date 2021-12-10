import random

import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 10, 14 # mean and standard deviation

def rdn_age():
    s = np.random.normal(mu, sigma, 100)
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()
    s.sort()
    for i in s:
        print(round(i))
def rdn_activites():
    s = np.random.normal(mu, sigma, 100)
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()
    s.sort()
    for i in s:
        print(round(i))

if __name__ == '__main__':
    rdn_activites()