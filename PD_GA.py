import numpy as np
import random
from main import global_vars

matrix_dim, w, payoff_M, epoch = global_vars(0)

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
    bin_index = str(opp_hist[0]) + str(opp_hist[1]) + str(opp_hist[2])
    int_index = int(bin_index, 2)
    new_strat = GA_strat[int_index]
    
    return new_strat
    
def GA(GA_stratList):
    population = GA_stratList
    count = 0
    for p in population:
        offspring = breeding(p, population)
        population[count,2:13] = offspring
        count += 1
    new_pop = mutation(population)
    new_pop[:, 13] = 0
    
    return new_pop
    
def sort_pop(population):
    tot_score = population[:, 13].sum()
    count = 0
    norm_score = np.zeros((len(population), 1))
    cumsum_score = np.zeros((len(population), 1))
    
    for i in population[:, 13]:
        norm_score[count] = i / tot_score
        # print(i, norm_score[count])
        count += 1
    norm_population = np.concatenate((population, norm_score), axis=1)
    sorted_population = norm_population[(-norm_population[:,14]).argsort()]
    cumsum_score[:,0] = np.cumsum(sorted_population[:,14])
    sorted_population = np.concatenate((sorted_population, cumsum_score), axis=1)
    # print('sp=', sorted_population)
    
    return sorted_population

def breeding(p, population):
    # print(x,y)
    x = p[0]
    y = p[1]
    neighbourhood = np.zeros((9,14))
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
                    h += 1
    # print(neighbourhood, '\n\n')
    sorted_neighbourhood = sort_pop(neighbourhood)  
    # print(sorted_neighbourhood, '\n\n')       
    breeders = selection(sorted_neighbourhood)
    offspring = SPcrossover(breeders)
    # print(offspring)
    return offspring
    
        
                
def selection(population):   
    breeders = np.zeros((2, 11), dtype=np.float)
    j = 0
    while j < 2:
        R = random.uniform(0,1) #beta.rvs(1, 6)
        # if sorted_population[0,7] < R <= 1:
        for k in range(1,len(population)):
            # print(R, population[k,15])
            if R < population[k,15]:
                breeders[j, :] = population[k-1, 2:13]
                # print(R, k)
                break
        j += 1
    # print(breeders)
    return breeders
    
def SPcrossover(breeders):
    R = np.random.randint(1, 11)
    offspring1 = np.concatenate((breeders[0,0:R], breeders[1, R:]))
    offspring2 = np.concatenate((breeders[1,0:R], breeders[0, R:]))
    # print('*K*K*', offspring1, offspring2) 
    return random.choice([offspring1, offspring2])
    
def mutation(population):
    for i in range(len(population)):
        R = random.uniform(0,1) 
        if R < 0.1:
            r = np.random.choice(range(2,14))
            if population[i,r] == 0:
                population[i,r] = 1
            elif population[i,r] == 1:
                population[i,r] = 0
    return population
