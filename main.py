import matplotlib.pyplot as plt
import time
import os
import numpy as np
from genetic import Model, Node
from functions import printProgressBar, cross_validation_split, select_validate


def preprocessData():
    file = open('./wdbc.data', 'r')
    data = []
    for row in file.readlines():
        line = row.split(',')
        line[len(line)-1] = line[len(line) - 1].split('\n')[0]
        dic = {
            'input': [],
            'desire_output': [],
        }
        dic['input'] = [float(d) for d in line[2:32]]
        if line[1] == 'M':
            dic['desire_output'].append(0)
        else:
            dic['desire_output'].append(1)
        data.append(dic)

    col_length = len(data[0]['input'])

    mydict = {}
    for col in range(col_length):
        mydict.update({f'{col}': []})

    for d in data:
        for col, val in enumerate(d['input']):
            mydict[f'{col}'].append(val)

    max_val = []
    for col in range(col_length):
        max_val.append(np.max(mydict[f'{col}']))

    for d in data:
        for col in range(col_length):
            d['input'][col] = d['input'][col] / max_val[col]

    return data


def buildLayer(node):
    return [Node() for i in range(node)]


def crossover(p1, p2):
    selected_chromosome = []

    for i in range(len(p1.chromosome)):
        rand = np.random.randint(0, 2)
        if rand == 0:
            selected_chromosome.append(p1.chromosome[i])
        else:
            selected_chromosome.append(p2.chromosome[i])

    child = Model()
    child.create(buildLayer(node=len(p1.inputLayer)), [buildLayer(node=len(layer))
                                                       for layer in p1.hiddenLayers], buildLayer(node=len(p1.outputLayer)))
    child.chromosome = np.array(selected_chromosome)
    child.updateNeuralNetwork()
    return child


def mutation(prototype):
    p_copy = prototype
    new_chromosome = []
    for l, layer in reversed(list(enumerate(p_copy.layers))):
        if l > 0:
            for node in layer:
                mutation_prob = np.random.randint(0, 2)
                if mutation_prob == 1:
                    for i, w in enumerate(node.w):
                        epsilon = np.random.uniform(-1, 1)
                        node.updateWeight(w + epsilon, i)

    for l, layer in reversed(list(enumerate(p_copy.layers))):
        if l > 0:
            for node in layer:
                for w in node.w:
                    new_chromosome.append(w)

    p_copy.chromosome = np.array(new_chromosome)

    return p_copy


def SUS(ranked_population):
    min = 0.8
    max = 2 - min

    n = len(ranked_population)
    for i, p in enumerate(ranked_population):
        pi = (min + ((max-min)*((i-1)/n)))/n
        p['pi'] = pi
        p['ni'] = n * pi

    ptr = np.random.uniform(0, 1)
    sum = 0
    selected_population = []
    for i, p in enumerate(ranked_population):
        sum += p['ni']
        while sum > ptr:
            selected_population.append(p)
            ptr += 1

    return selected_population


data = preprocessData()


inputLayer_n = 30
outputLayer_n = 1

# print(
#     f'neural shape = {inputLayer_n} - {[node for node in hiddenNode_n]} - {outputLayer_n}')


population_n = 50
population = []
for i in range(population_n):
    inputLayer = buildLayer(node=inputLayer_n)
    hiddenLayers = [buildLayer(node=7)
                    for i in range(1)]
    outputLayer = buildLayer(node=outputLayer_n)

    p = Model()
    p.create(inputLayer, hiddenLayers, outputLayer)
    population.append(p)


t_max = 50
global_SSE = []

cross_data = cross_validation_split(cross_validate_num=0.1, dataset=data)
block = cross_data["data_block"]
rand_set = cross_data["rand_set"]
reminder_set = cross_data["rem_set"]


def clear(): return os.system('clear')


cross_validation_plot = []
for c in range(10):
    res = select_validate(block, rand_set, c, reminder_set)
    train = res["train"]
    cross_valid = res["cross_valid"]
    printProgressBar(
        0, t_max, prefix=f'crossvalidation : {c+1}', printEnd='\n', length=50)
    best = []
    for t in range(t_max):
        ranked_based = []
        printProgressBar(0, len(population),
                         prefix='evaluating...', length=25)
        for i, p in enumerate(population):
            SSE = p.evaluate(train)
            dict_p = {
                "SSE": SSE,
                "Model": p
            }
            ranked_based.append(dict_p)
            printProgressBar(i+1, len(population),
                             prefix=f'evaluating... SSE={SSE}', length=25)
        clear()

        # selction process
        ranked_based = sorted(ranked_based, key=lambda k: k['SSE'])
        mating_pool = SUS(ranked_based)
        # mating process
        next_gen = []
        mating_count = 1
        while mating_count <= len(mating_pool)*0.7:
            parents = []
            for i in range(2):
                rand = np.random.randint(0, len(mating_pool))
                parents.append(mating_pool[int(rand)])
            # crossover
            child1 = crossover(parents[0]['Model'], parents[1]['Model'])
            next_gen.append(child1)
            # mutation
            child2 = mutation(prototype=child1)
            next_gen.append(child2)
            mating_count += 1

        # select and mutate from mating pool
        while len(next_gen) < population_n:
            rand = np.random.randint(0, len(mating_pool))
            pool_p = mating_pool[int(rand)]
            child = mutation(prototype=pool_p['Model'])
            next_gen.append(child)

        printProgressBar(
            t+1, t_max, prefix=f'evolutioning.. t : {t+1} best : {mating_pool[len(mating_pool)-1]["SSE"]}', suffix=f'cross validate : {c+1}', length=50, printEnd='\n')
        best.append(mating_pool[0]["SSE"])
        population = next_gen
    print('best in all gen')
    print(np.max(best))
    time.sleep(3)

    # cross validation
    printProgressBar(0, len(population),
                     prefix=f'cross validationing... {c+1}', length=25)
    sse_cross = []
    for i, p in enumerate(population):
        SSE = p.evaluate(cross_val)
        sse_cross.append(SSE)
        printProgressBar(i+1, len(population),
                         prefix=f'cross validationing... {c+1}', length=25)
    cross_validation_plot.append(np.max(sse_cross))
    clear()

plt.plot(cross_validation_plot)
plt.title('cross validation')
plt.show()
