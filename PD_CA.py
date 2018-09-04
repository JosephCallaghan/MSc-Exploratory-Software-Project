# ============================================================================ #
# PD_CA.py - This script contains the functions that maintain and update the cellular
# automaton for the PD game:
#           * initial_pop() - generates the initial population in the lattice
#           * update_strat() - updates the current strategy to be played next round
#                               for both players of the game
#           * score_func() - calculates the awarded payoff to each player
#           * UPDATE() - calls upon the above functions to perform the update of 
#                           the automaton
#           * getHistory() - retrieves the history of an individual in the automaton
#                               depending on their strategy-set
#           * IndexBCs1() - returns correct values to apply periodic boundary
#                           conditions for a 5x5 grid of cells
#           * IndexBCs2() - returns correct values to apply periodic boundary
#                           conditions for a 3x3 grid of cells
#
# KEY VARIABLES:
#       GA_stratList - see main.py
#       G_temp - temporary refined GA_stratList that contains only the GA agents 
#                required for the update in order to quicken computation
#       pop_p1 - the strategy-set indicator of player 1
#       pop_p2 - the strategy-set indicator of player 2
#       p1cs/p1ns - player 1's current/new strategy, respectively
#       p2cs/p2ns - player 2's current/new strategy, respectively
#       p1_hist - player 1's strategy choice history from the past 3 rounds
#       p2_hist - player 2's strategy choice history from the past 3 rounds
# 
# Last edited: 01/09/2018
#
# ============================================================================ #

import numpy as np
from collections import deque
import random
from main import global_vars
from PD_GA import GA_Strat

matrix_dim, w, payoff_M, epoch = global_vars(0)
# Produce a binary string of length y
binary_list = lambda y: [np.random.randint(0,1) for x in range(1,y+1)]

def initialPop(strats, probs):
    # Generate randomly distributed population of strategy-sets defined in main.py
    # with frequency 'probs'
    population = np.zeros([matrix_dim, matrix_dim], dtype=np.int8)
    for i in range(np.square(matrix_dim)):
        np.put(population, [i], [np.random.choice(strats, p=probs)])
    
    # Count the number of GA agents (5's) to produce the list of GA agents (GA_stratList)
    num_fives = np.count_nonzero(population == 5)
    GA_stratList = np.zeros((num_fives, 14), dtype = np.int32)
    currentStrats = population.copy()
    count = 0
    
    # Populate the CS-array (currentStrats) with an intitial strategy choice (either
    # 0 or 1) depnding on the strategy-set in the population array.
    for i in range(matrix_dim):
        for j in range(matrix_dim):
            if currentStrats[i, j] == 2: # If TFT, start with cooperate (1)
                currentStrats[i, j] = 1
                 
            if currentStrats[i, j] == 3: # If a fixed strategy produced by a GA only simulation, initialise by indexing an initial opponents history (p2hist)
                fixed_strat = np.array([0,0,0,0,0,0,1,1]) 
                p2hist = np.random.choice([0, 1], size=(3,))
                bin_index = str(p2hist[0]) + str(p2hist[1]) + str(p2hist[2])
                int_index = int(bin_index, 2)
                currentStrats[i, j] = fixed_strat[int_index]
                
            if population[i,j] == 5:    # If a GA agent, generate initial random sample and add to GA_stratList
                GA_index = np.array([i, j]) # Obtain GA agent index
                GA_string = np.random.choice([0, 1], size=(11,)) # Generate random genotype
                GA_score = np.zeros(1) # Designate a bit for its score
                GA_strat = np.concatenate((GA_string, GA_score))
                GA_stratList[count,:] = np.concatenate((GA_index, GA_strat)) # Concatenate all information
                p2hist = np.random.choice([0, 1], size=(3,))
                bin_index = str(p2hist[0]) + str(p2hist[1]) + str(p2hist[2])
                int_index = int(bin_index, 2)
                currentStrats[i, j] = GA_strat[int_index] # Generate and initial strategy choice from its genotype and random initial p2hist
                count += 1
    print(population, currentStrats)
    return population, currentStrats, GA_stratList

def UPDATE(population, currentStrats, GA_stratList, av_scores):
    # Randomly select a cell to update
    x = np.random.randint(matrix_dim)
    y = np.random.randint(matrix_dim)
    scores = np.zeros([matrix_dim, matrix_dim], dtype=np.int32) # Generate matrix to store which individual scored what
    
    # Obtain indexes of the surrounding 5x5 grid of cells required to perform the update    
    xp1 = x+1
    xp2 = x+2
    xm1 = x-1
    xm2 = x-2
    yp1 = y+1
    yp2 = y+2
    ym1 = y-1
    ym2 = y-2
    
    # Apply periodic boundary conditions (BCs)
    xp1, xp2, xm1, xm2, yp1, yp2, ym1, ym2 = indexBCs1(xp1, xp2, xm1, xm2, yp1, yp2, ym1, ym2, matrix_dim)
    
    # Redefine indexes of the surrounding 5x5 grid of cells after BCs
    mooreX3 = np.array([xm2, xm1, x, xp1, xp2])
    mooreY3 = np.array([ym2, ym1, y, yp1, yp2])
    count1 = 0
    count2 = 0
    G_temp = np.zeros(1) # G_temp will be the refined GA-stratList of 25 (5x5) or less GA agents required for the update
    
    # Loop through the GA_stratList to extract G_temp 
    for p in GA_stratList:
        for i in mooreX3:
            for j in mooreY3:
                if (p[:2] == [i,j]).all() == True and count1 == 0:
                    G_temp = np.array([p])
                    GA_stratList = np.delete(GA_stratList,(count2), axis=0)
                    count1 += 1
                    count2 -= 1
                elif (p[:2] == [i,j]).all() == True:
                    G_temp = np.concatenate((G_temp,[p]), axis=0)
                    GA_stratList = np.delete(GA_stratList,(count2), axis=0)
                    count2 -= 1
        count2 += 1
    av_numgames = 0
    
    # Each cell within the updated cell's moore neighbourhood is selected sequentially 
    # to calculate its score
    for i in range (-1, 2):
        for j in range(-1,2):
            score = 0
            xpi = x+i
            xpip1 = xpi+1
            xpim1 = xpi-1
            ypj = y+j
            ypjp1 = ypj+1
            ypjm1 = ypj-1
            
            xpi, xpip1, xpim1, ypj, ypjp1, ypjm1 = indexBCs2(xpi, xpip1, xpim1, ypj, ypjp1, ypjm1, matrix_dim)
            
            # Define moore neighbourhood for cell we are scoring
            mooreX1 = np.array([xpi, xpim1, xpim1, xpim1, xpi, xpip1, xpip1, xpip1])
            mooreY1 = np.array([ypjm1, ypjm1, ypj, ypjp1, ypjp1, ypjp1, ypj, ypjm1])
                                    
            pop_p1 = population[xpi, ypj]
            p1cs = currentStrats[xpi, ypj]
            
            # Define the histroy of player 1 depending on its strategy-set and store in 
            # a double ended queue
            if pop_p1 == 5:
                p1hist = getHistory(xpi, ypj, G_temp)
                p1hist.pop()
                p1hist.appendleft(p1cs)
            elif pop_p1 == 0:
                p1hist = deque(list([0, 0, 0]))
            elif pop_p1 == 1:
                p1hist = deque(list([1, 1, 1]))
            elif pop_p1 == 2 or pop_p1 == 3:
                p1hist = deque([1,1,1])
            
            # Loop through the cells in the scoring cell's moore neighbourhood which will be
            # player 1's opponents
            for k in range(8):
                # Define the histroy of player 2 depending on its strategy-set and store in 
                # a double ended queue
                pop_p2 = population[mooreX1[k], mooreY1[k]]
                p2cs = currentStrats[mooreX1[k], mooreY1[k]]
                if pop_p2 == 5:
                    p2hist = getHistory(mooreX1[k], mooreY1[k], G_temp)
                    p2hist.pop()
                    p2hist.appendleft(p2cs)
                elif pop_p2 == 0:
                    p2hist = deque(list([0, 0, 0]))
                elif pop_p2 == 1:
                    p2hist = deque(list([1, 1, 1]))
                elif pop_p2 == 2 or pop_p2 == 3:
                    p2hist = deque([1, 1, 1])     
                    
                # Calculate the awarded payoff after playing the iterated PD game
                score1, c, p1hist = scoreFunc(pop_p1,pop_p2, p1hist, p2hist, [xpi, ypj], [mooreX1[k], mooreY1[k]], G_temp)#, obj_func)
                score += score1
                av_numgames += c
                
                count = 0  
                # If player is a GA player, update it's history   
                if (G_temp == 0).all() == False:
                    for g in GA_stratList:
                        if (p[:2] == [i,j]).all() == True and count1 == 0: 
                            G_temp[count,2:5] = np.array(p1hist)
                    count += 1
            
            # Maintain average scores matrix (see main.py)
            for n in range(4):
                if n == pop_p1:
                    av_scores[0,n] += 1
                    row = int(av_scores[0,n])
                    av_scores[row,n] = score/8
                    break
                    
                if pop_p1 == 5:
                    av_scores[0,4] += 1
                    row = int(av_scores[0,4])
                    av_scores[row,4] = score/8
                    break
                    
            scores[xpi,ypj] = score # Update sparse score matrix of cells in moore neighbourhood
            av_numgames = av_numgames/8
    
    # Find the individual that scored the highest in the updated cell's moore neighbhourhood
    local_scores = np.array([scores[x, ym1], scores[xm1, ym1], scores[xm1, y], scores[xm1, yp1], scores[x, yp1], scores[xp1, yp1], scores[xp1, y], scores[xp1, ym1], scores[x,y]])
    D = np.where(local_scores == local_scores.max())    
    
    # If individual is a GA agent, add score to G_temp as the individuals fitness
    if (G_temp == 0).all() == False: 
        for p in G_temp:
            for i in mooreX3[1:4]:
                for j in mooreY3[1:4]:
                    if(p[:2] == [i,j]).all() == True:
                        p[13] += scores[p[0], p[1]]
    
    # If two cells are tied for the winner, pick one at random    
    if np.size(D) > 1:
        D = random.choice(D[0])
    else:
        D = int(D[0])
    
    mooreX2 = np.array([x, xm1, xm1, xm1, x, xp1, xp1, xp1])
    mooreY2 = np.array([ym1, ym1, y, yp1, yp1, yp1, y, ym1])
    
    
    if D < 8:    # If the updated cell's strategy needs updating
        for k in range(8):
            if D == k:
                OLD = population[x,y]
                population[x,y] = population[mooreX2[k], mooreY2[k]]    # Update cell with new strategy in the population
                if population[mooreX2[k], mooreY2[k]] == 5:     # If the winning strategy IS a 5
                    count = 0
                    for i in G_temp:                  # Loop through list of GA agents
                        if OLD == 5 and (i[:2] == np.array([x,y])).all() == True:     # If updated cell WAS a 5, store its index
                            for j in G_temp:          # Loop through list again
                                if(j[:2] == np.array([mooreX2[k], mooreY2[k]])).all() == True:    # Find the genetic code of the winning strategy
                                    G_temp[count,2:] = j[2:]     # Update cell to adopt the winning strategy's genetic code
                                    currentStrats[x, y] = currentStrats[mooreX2[k], mooreY2[k]] # Update its current strategy
                            break
                        
                        if OLD != 5:    # If updated cell WASN'T a 5
                            newGA_index = np.array([x, y])
                            newGA_strat = i[2:]
                            newGA_agent = np.concatenate((newGA_index, newGA_strat))
                            G_temp = np.concatenate((G_temp, np.array([newGA_agent])), axis=0)    # Copy new strategy and add it to GA agent list
                            currentStrats[x, y] = currentStrats[mooreX2[k], mooreY2[k]]
                            break
                        count += 1
                        
                if population[mooreX2[k], mooreY2[k]] != 5:     # If the winning strategy ISN'T a 5
                    currentStrats[x,y] = population[x,y]
                    if population[x,y] == 2 or population[x, y] == 3:   # Update current strategy array accordingly
                        currentStrats[x,y] = 1
                    if OLD == 5:            # If updated cell WAS a 5
                        count = 0
                        for i in G_temp: # Find it in G_temp
                            if (i[:2] == np.array([x,y])).all() == True:
                                break
                            count += 1
                        if count < len(G_temp):
                            G_temp = np.delete(G_temp,(count), axis=0) # Delete the losing strategy from the list
                break
       
    # Rebuild the full GA_stratList         
    if (G_temp == 0).all() == False:       
        GA_stratList = np.concatenate((GA_stratList, G_temp), axis=0)
                
    return population, currentStrats, GA_stratList, av_scores, av_numgames


def updatestrat(pop_p1, pop_p2, p1hist, p2hist, p1_coord, p2_coord, GA_stratList):
    p1cs = p1hist[0]
    p2cs = p2hist[0]
    
    # Determine the new strategy based on the individuals strategy-set (see main.py)
    if pop_p1 == 2:
        p1ns = p2cs
    elif pop_p1 == 3:
        fixed_strat = np.array([0,0,0,0,0,0,1,1]) 
        bin_index = str(p2hist[0]) + str(p2hist[1]) + str(p2hist[2])
        int_index = int(bin_index, 2)
        p1ns = fixed_strat[int_index]
    elif pop_p1 == 5:
        p1ns = GA_Strat(p2hist, GA_stratList, p1_coord)
    else:
        p1ns = p1cs
        
    if pop_p2 == 2:
        p2ns = p1cs
    elif pop_p2 == 3:
        fixed_strat = np.array([0,0,0,0,0,0,1,1]) 
        bin_index = str(p1hist[0]) + str(p1hist[1]) + str(p1hist[2])
        int_index = int(bin_index, 2)
        p2ns = fixed_strat[int_index]
    elif pop_p2 == 5:
        p2ns = GA_Strat(p1hist, GA_stratList, p2_coord)
    else:
        p2ns = p2cs
    
    # Individual may make a mistake with a probability 0.01
    R = random.uniform(0,1)
    if R < 0.01:
        if p1ns == 1:
            p1ns = 0
        else:
            p1ns = 1
            
    return p1ns, p2ns

def scoreFunc(pop_p1, pop_p2, p1hist, p2hist, p1_coord, p2_coord, GA_stratList):
    score = 0
    p1ns = p1hist[0]
    p2ns = p2hist[0]
    score += payoff_M[p1ns, p2ns]   # Calculate the score using the PD payoff matrix
    c = 1
    
    # Play the iterated game with probability of playing another game of w (see main.py)
    for m in range(100):
        R = random.uniform(0,1)
        if R > w:
            break 
        # Update strategy of each player after each round
        p1ns, p2ns = updatestrat(pop_p1, pop_p2, p1hist, p2hist, p1_coord, p2_coord, GA_stratList)
        score += payoff_M[p1ns, p2ns]
        
        # Update history of each player each round
        p1hist.pop()
        p1hist.appendleft(p1ns)
        p2hist.pop()
        p2hist.appendleft(p2ns)
        c += 1
    
    # Calculate average score per round
    score = score/c
    return score, c, p1hist


    
def getHistory(x, y, G_temp):
    for p in G_temp:
        if(p[:2] == [x, y]).all() == True:
            p_hist = deque(list([p[2], p[3], p[4]]))
            break
    return p_hist

def indexBCs1(x1, x2, x3, x4, y1, y2, y3, y4, matrix_dim):
    if x1 == matrix_dim:
        x1 = 0
        x2 = 1
    if x2 == matrix_dim:
        x2 = 0  
    if x3 == -1:
        x3 = matrix_dim-1
        x4 = matrix_dim-2
    if x4 == -1:
        x4 = matrix_dim-1
    
    if y1 == matrix_dim:
        y1 = 0
        y2 = 1
    if y2 == matrix_dim:
        y2 = 0  
    if y3 == -1:
        y3 = matrix_dim-1
        y4 = matrix_dim-2
    if y4 == -1:
        y4 = matrix_dim-1
    return x1, x2, x3, x4, y1, y2, y3, y4

def indexBCs2(x1, x2, x3, y1, y2, y3, matrix_dim):
    if x1 == matrix_dim:
        x1 = 0
        x2 = 1
    if x2 == matrix_dim:
        x2 = 0  
    if x1 == -1:
        x1 = matrix_dim-1
        x3 = matrix_dim-2
    if x3 == -1:
        x3 = matrix_dim-1
    
    if y1 == matrix_dim:
        y1 = 0
        y2 = 1
    if y2 == matrix_dim:
        y2 = 0  
    if y1 == -1:
        y1 = matrix_dim-1
        y3 = matrix_dim-2
    if y3 == -1:
        y3 = matrix_dim-1
    return x1, x2, x3, y1, y2, y3