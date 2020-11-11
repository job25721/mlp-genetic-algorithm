import numpy as np


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def cross_validation_split(cross_validate_num, dataset):
    cross_len = int(round((len(dataset) * cross_validate_num), 0))
    block = []
    random_set = []
    reminder_set = []
    for i in range(10):
        rand = np.random.randint(0, 10)
        while random_set.__contains__(rand):
            rand = np.random.randint(0, 10)
        random_set.append(rand)
        block.append(dataset[i * cross_len:cross_len + (i * cross_len)])
        if i == 9 and sum([len(b) for b in block]) < len(dataset):
            reminder_set = dataset[cross_len +
                                   (i * cross_len):len(dataset)]
    return {
        "data_block": block,
        "rand_set": random_set,
        "rem_set": reminder_set
    }


def select_validate(block, random_set, c, rem_set):
    cross_valid = block[random_set[c]]
    train_idx = random_set.copy()
    train_idx.remove(random_set[c])
    train = []
    for n in range(9):
        train += block[train_idx[n]]
        if c == 9 and n == 8:
            train += rem_set
    return {
        "train": train,
        "cross_valid": cross_valid
    }
