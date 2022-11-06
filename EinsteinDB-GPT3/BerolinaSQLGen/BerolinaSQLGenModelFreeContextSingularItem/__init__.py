# -*- coding: utf-8 -*-
#
# """
# Ornstein–Uhlenbeck process:
# https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
# """
# import numpy as np
# Compare this snippet from EinsteinDB-GPT3/einstAI-DALLE2/__init__.py:

# """

from ddpg import DDPG
import replay_memory
import gym
from itertools import count
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import time
import datetime
import argparse
import random
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import os

from AML.Synthetic.naru.models import DDPG
from AML.Synthetic.naru.replay_memory import ReplayMemory, Transition

from EINSTAI.OUCausetFlowProcess.PGUtils import BerolinaSQLGenDQNWithBoltzmannNormalizer


def main():
    filelist = ["qnoether.txt", "random.txt"]
    metric_name = "throughput"
    draw_lines(filelist, metric_name)

__all__ = ["DDPG", "BerolinaSQLGenDQNWithBoltzmannNormalizer", "ReplayMemory", "Transition"]



# -*- coding: utf-8 -*-





def draw_lines(filelist, metric_name):

    # filelist = ["qnoether.txt", "random.txt"]
    # metric_name = "throughput"
    # draw_lines(filelist, metric_name)

    """
    Draw lines for the metrics in the filelist.
    :param filelist: list of file names
    :param metric_name: name of the metric
    :return:
    """

    # filelist = ["qnoether.txt", "random.txt"]
    # metric_name = "throughput"
    # draw_lines(filelist, metric_name)


    # filelist = ["qnoether.txt", "random.txt"]
    # metric_name = "throughput"
    for filename in filelist:
        # print(filename)
        # print(metric_name)
        df = pd.read_csv(filename, sep=",")
        # print(df)
        plt.plot(df["step"], df[metric_name], label=filename)

    plt.legend()
    plt.show()

# filelist = ["qnoether.txt", "random.txt"]


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
#



