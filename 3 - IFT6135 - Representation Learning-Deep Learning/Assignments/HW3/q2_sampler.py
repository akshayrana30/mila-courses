import numpy as np
import random


def distribution1(x, batch_size=512):
    # Distribution defined as (x, U(0,1)). Can be used for question 2.3
    # USAGE:
    #     sampler = iter(samplers.distribution1(0))
    #     data = next(sampler)
    while True:
        yield(np.array([(x, random.uniform(0, 1), ) for _ in range(batch_size)]))
