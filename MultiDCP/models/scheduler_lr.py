

def step_lr(step_pattern, step):
    if step >= step_pattern[3]:
        return 1
    elif step >= step_pattern[2]:
        return 1
    elif step >= step_pattern[1]:
        return 10
    else:
        return 100
