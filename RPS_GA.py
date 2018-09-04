import numpy as np
import random
from main import global_vars

matrix_dim, w, payoff_M, epoch = global_vars(1)

def GA_Strat(opp_hist, GA_stratList, p1_coord):
    if p1_coord[0] == -1:
        p1_coord[0] = matrix_dim-1
    elif p1_coord[0] == -2:
        p1_coord[0] = matrix_dim-2
    if p1_coord[1] == -1:
        p1_coord[1] = matrix_dim-1
    elif p1_coord[1] == -2:
        p1_coord[1] = matrix_dim-2

    for i in GA_stratList:
        if(i[:2] == np.array(p1_coord)).all().any() == True:
            GA_strat = i[2:]
            break
   
    base3_index = str(opp_hist[0]) + str(opp_hist[1])
    int_index = int(base3_index, 3)
    new_strat = GA_strat[int_index]
    
    return new_strat
    
def GA(GA_stratList):
    population = GA_stratList
    # print('pop=', population)
    # sorted_pop = sort_pop(population)
    # num_breeders = int(2 * round(float(0.4*len(population))/2)
    count = 0
    for p in population:
        offspring = breeding(p, population)
        population[count,2:14] = offspring
        count += 1
    new_pop = mutation(population)
    new_pop[:, 14] = 0
    
    return new_pop
    
def sort_pop(population):
    tot_score = population[:, 14].sum()
    count = 0
    norm_score = np.zeros((len(population), 1))
    cumsum_score = np.zeros((len(population), 1))
    
    for i in population[:, 14]:
        norm_score[count] = i / tot_score
        # print(i, norm_score[count])
        count += 1
    norm_population = np.concatenate((population, norm_score), axis=1)
    sorted_population = norm_population[(-norm_population[:,15]).argsort()]
    cumsum_score[:,0] = np.cumsum(sorted_population[:,15])
    sorted_population = np.concatenate((sorted_population, cumsum_score), axis=1)
    # print('sp=', sorted_population)
    
    return sorted_population

def breeding(p, population):
    # print(x,y)
    x = p[0]
    y = p[1]
    neighbourhood = np.zeros((9,15))
    h = 0
    for q in population:
        for i in range(x-1,x+2):
            for j in range(y-1,y+2):
                if i == matrix_dim:
                    i = 0
                if i == -1:
                    i = matrix_dim-1
                if j == matrix_dim:
                    j = 0
                if j == -1:
                    j = matrix_dim-1
                
                if (q[:2] == [i,j]).all() == True:
                    neighbourhood[h, :] = q
    # print(neighbourhood, '\n\n')
    sorted_neighbourhood = sort_pop(neighbourhood)  
    # print(sorted_neighbourhood, '\n\n')       
    breeders = selection(sorted_neighbourhood)
    offspring = SPcrossover(breeders)
    # print(offspring)
    return offspring
        
                
def selection(population):   
    breeders = np.zeros((2, 12), dtype=np.float)
    j = 0
    while j < 2:
        R = random.uniform(0,1) #beta.rvs(1, 6)
        # if sorted_population[0,7] < R <= 1:
        for k in range(1,len(population)):
            # print(R, sorted_population[k,16])
            if R < population[k,16]:
                breeders[j, :] = population[k-1, 2:14]
                # print(R, k)
                break
        j += 1
    return breeders
    
def SPcrossover(breeders):
    R = np.random.randint(1, 12)
    offspring1 = np.concatenate((breeders[0,0:R], breeders[1, R:]))
    offspring2 = np.concatenate((breeders[1,0:R], breeders[0, R:]))
    return random.choice([offspring1, offspring2])
    
def mutation(population):
    for i in range(len(population)):
        R = random.uniform(0,1)
        if R < 0.1:
            r = np.random.choice(range(2,15))
            if population[i,r] == 0:
                population[i,r] = np.random.choice([1,2])
            elif population[i,r] == 1:
                population[i,r] = np.random.choice([0,2])
            elif population[i,r] == 2:
                population[i,r] = np.random.choice([0,1])                 
    return population
