
import numpy as np

def repair(args,resolving_set,ntable):
    temp_set = resolving_set.copy()
    p = np.random.choice(range(len(temp_set)))
    temp_set.pop(p)
    temp_panel = []
    for index, value in enumerate(temp_set):
        if index == 0:
            temp_panel = ntable[value]
        else:
            temp_panel = temp_panel.intersection(ntable[value])

    if len(temp_panel)==0:
        resolving_set = temp_set

    return resolving_set

def repair_iter(args,resolving_set,ntable,iter):

    for i in range(iter):
        print(i)
        resolving_set = repair(args,resolving_set, ntable)
    return resolving_set,len(resolving_set)

