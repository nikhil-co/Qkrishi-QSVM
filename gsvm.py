from deap import base, creator, tools, algorithms
import fitness
import random
import numpy as np

def gsvm(nqubits, depth, nparameters, X, y,
         mu=100, lambda_=150, cxpb=0.7, mutpb=0.3, ngen=2000,
         use_pareto=True, verbose=True, weights=[-1.0,1.0],
         debug=True):
    bits_puerta = 5
    long_cadena = depth * nqubits * bits_puerta
    creator.create("FitnessMulti", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness = creator.FitnessMulti, statistics=dict)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("Individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, long_cadena)
    toolbox.register("Population", tools.initRepeat, list, toolbox.Individual)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.2)
    toolbox.register('select', tools.selNSGA2)
    toolbox.register("evaluate", fitness.Fitness(nqubits, nparameters, X, y, debug=debug))

    pop = toolbox.Population(n=mu)

    stats_wc = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_wc.register('media',np.mean)
    stats_wc.register('std',np.std)
    stats_wc.register('max',np.max)
    stats_wc.register('min',np.min)
    stats_acc = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats_acc.register('media',np.mean)
    stats_acc.register('std',np.std)
    stats_acc.register('max',np.max)
    stats_acc.register('min',np.min)
    mstats = tools.MultiStatistics(wc=stats_wc, acc=stats_acc)

    logbook = tools.Logbook()
    pareto = tools.ParetoFront(similar = np.array_equal)
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox,
                                             mu, lambda_, cxpb, mutpb, ngen,
                                             stats=mstats,
                                             halloffame=pareto, verbose=verbose)
    pareto.update(pop)
    return pop, pareto, logbook
