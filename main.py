# ============================================================================ #
# main.py - The simulations are run from this script. It calls upon the other
# scripts:
#           * PD_CA - The cellular automaton functions for the PD game
#           * RPS_CA - The cellular automaton functions for the RPS game
#           * PD_GA - The genetic algorithm functions for the PD game
#           * RPS_GA - The genetic algorithm functions for the RPSgame
#
# This script produces the visual representation of the automaton lattice and 
# outputs the diagnostic text files:
#           * freq.out - The number of each strategy-set agent per epoch
#           * av_scores.out - The average fitness scores for each strategy-set
#                             per epoch
#           * best_scores.out - The best fitness scores for each strategy-set
#                               per epoch
#           * GA_stratList.out - The GA strategy-sets genotypes produced for all
#                            GA agents remaining at the end of the simulation
#
# KEY VARIABLES:
#       new_pop - the P-array (see report) is the 2D array acting as the lattice
#                   and stores the strategy-set identification
#       currentStrats - the CS-array (see report) is the 2D array that stores
#                       current strategy of each strategy-set
#       GA_stratList - the maintained list conataining all information of GA 
#                       agents in the format 'coordinates|genotype|score'
# 
# Last edited: 01/09/2018
#
# ============================================================================ #


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
np.set_printoptions(precision=1, suppress=True)

# global_vars defines the global variables such as the matrix dimenions, probability 
# that another game will be played and epoch length, as well as defining the payoff 
# matrices. The game choice indicates which game, PD or RPS, is being simulated.
def global_vars(gamechoice):
    matrix_dim = 50
    w = 0
    epoch = np.square(matrix_dim)
    
    if gamechoice == 0:
        payoff_M = np.array([[1, 5], [0, 3]])
    elif gamechoice == 1:
        payoff_M = np.array([[1, 0, 3], [3, 1, 0], [0, 3, 1]]) 
    return matrix_dim, w, payoff_M, epoch

import RPS_CA
import PD_CA
matrix_dim, w, payoff_M, epoch = global_vars(0)

if __name__ == "__main__":
    TYPE = 'RPS'                             # 'PD' or 'RPS' defines which game to be simulated
    its = epoch*100                         # Number of iterations to be performed
    pd_strats = np.array([0, 1])            # Define the strategies to be included in simulation for PD game [0=Def, 1 =Coop, 2=TFT, 3=fixed_GA 5=GA]
    pd_freq = [0.5, 0.5]                    # Defines the initial percentage of each strategy for the PD game
    rps_strats = np.array([0,1,2])          # Define the strategies to be included in simulation for RPS game [0=R, 1=P, 2=S, 3=NE, 4=CT, 5=GA]
    rps_freq = [0.33, 0.33, 0.34]           # Defines the initial percentage of each strategy for the RPS game
    
    # Defines matrices to be printed to the output files for either the PD or RPS game
    if TYPE == 'RPS':
        from RPS_GA import GA
        pre = 'RPS_CA'
        num_rock = np.zeros(its+1, dtype=np.int64)         # 'num_*' keeps track of the strategy-set frequencies
        num_paper = np.zeros(its+1, dtype=np.int64)
        num_scissor = np.zeros(its+1, dtype=np.int64)
        num_NE = np.zeros(its+1, dtype=np.int64)
        num_CC = np.zeros(its+1, dtype=np.int64)
        num_GA = np.zeros(its+1, dtype=np.int64)
        cont_avScore = np.zeros((int(its/epoch)+1,6), dtype=np.float)   # stores average score per iteration (row=iteration, column=strategy-set)
        cont_bestScore = np.zeros((int(its/epoch)+1,6), dtype=np.float)     # stores best score per iteration  (row=iteration, column=strategy-set)
        freq_densities = np.zeros((its+1, 6), dtype=np.int32)           # stores strategy freqeuncies  (row=iteration, column=strategy-set)
        av_scores = np.zeros((epoch+1, 6), dtype=np.float64)            # stores average score per epoch  (row=epoch, column=strategy-set)
        new_pop, currentStrats, GA_stratList = RPS_CA.initialPop(rps_strats, rps_freq) # Generate the initial population
                
    elif TYPE == 'PD':
        from PD_GA import GA
        pre = 'PD_CA'
        num_coop = np.zeros((its+1,1), dtype=np.int64)  
        num_def = np.zeros((its+1,1), dtype=np.int64)
        num_TFT = np.zeros((its+1,1), dtype=np.int64)
        num_GTFT = np.zeros((its+1,1), dtype=np.int64)
        num_GA = np.zeros((its+1,1), dtype=np.int64)
        cont_avScore = np.zeros((int(its/epoch)+1,5), dtype=np.float)
        cont_bestScore = np.zeros((int(its/epoch)+1,5), dtype=np.float)
        freq_densities = np.zeros((its+1, 5), dtype=np.int32)
        av_scores = np.zeros((epoch+1, 5), dtype=np.float64)
        new_pop, currentStrats, GA_stratList = PD_CA.initialPop(pd_strats, pd_freq)
    
    # Define colour map for lattice visual
    cmap = ListedColormap(['r', 'b', 'g'])
    
    # Initilise counters
    count = 0
    k = 0
    stop = 0
    av_numgames1 = 0
    
    # This loop performs the updates of the lattice, scores and frequencies
    for m in range(its+1):
        # Perform an update using asynchronous  random updating
        new_pop, currentStrats, GA_stratList, av_scores, av_numgames = eval(pre).UPDATE(new_pop, currentStrats, GA_stratList, av_scores)
        av_numgames1 += av_numgames
            
        # Plots the lattice state every epoch
        if m%epoch == 0:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
            cax = ax.matshow(new_pop,cmap=cmap)
            plt.show()
            # fig.savefig("{0:03d}.png".format(k), bbox_inches='tight', transparent=True)
            
            print('epoch =', k)
            if k > 0:
                GA_stratList = GA(GA_stratList)     # Perform the genetic algorithm on all GA agents per epoch
            k += 1   
            av_numgames1 = av_numgames1/epoch
            av_numgames1 = 0
     
        # Maintain freqeuncy and fitness matrices          
        if TYPE == 'RPS':
            freq_densities[m,0] = np.count_nonzero(new_pop == 0)
            freq_densities[m,1] = np.count_nonzero(new_pop == 1)
            freq_densities[m,2] = np.count_nonzero(new_pop == 2)
            freq_densities[m,3] = np.count_nonzero(new_pop == 3)
            freq_densities[m,4] = np.count_nonzero(new_pop == 4)
            freq_densities[m,5] = np.count_nonzero(new_pop == 5)
            
            if m%epoch == 0:
                if (rps_strats == 0).any() == True:
                    cont_avScore[count,0] = np.sum(av_scores[1:,0])/av_scores[0,0]
                    cont_bestScore[count,0] = av_scores[1:,0].max()
                if (rps_strats == 1).any() == True:   
                    cont_avScore[count,1] = np.sum(av_scores[1:,1])/av_scores[0,1] 
                    cont_bestScore[count,1] = av_scores[1:,1].max()            
                if (rps_strats == 2).any() == True:
                    cont_avScore[count,2] = np.sum(av_scores[1:,2])/av_scores[0,2]
                    cont_bestScore[count,2] = av_scores[1:,2].max() 
                if (rps_strats == 3).any() == True:
                    cont_avScore[count,3] = np.sum(av_scores[1:,3])/av_scores[0,3]
                    cont_bestScore[count,3] = av_scores[1:,3].max()
                if (rps_strats == 4).any() == True:
                    cont_avScore[count,4] = np.sum(av_scores[1:,4])/av_scores[0,4]
                    cont_bestScore[count,4] = av_scores[1:,4].max()
                if (rps_strats == 5).any() == True:
                    cont_avScore[count,5] = np.sum(av_scores[1:,5])/av_scores[0,5]
                    cont_bestScore[count,5] = av_scores[1:,5].max()
                av_scores = np.zeros((its*9, 6), dtype=np.float64)
                count += 1

        elif TYPE == 'PD': 
            freq_densities[m,0] = np.count_nonzero(new_pop == 0)
            freq_densities[m,1] = np.count_nonzero(new_pop == 1)
            freq_densities[m,2] = np.count_nonzero(new_pop == 2)
            freq_densities[m,3] = np.count_nonzero(new_pop == 3)
            freq_densities[m,4] = np.count_nonzero(new_pop == 5)
            
            if m%epoch == 0:
                if (pd_strats == 0).any() == True:
                    cont_avScore[count,0] = np.sum(av_scores[1:,0])/av_scores[0,0]
                    cont_bestScore[count,0] = av_scores[1:,0].max()
                if (pd_strats == 1).any() == True:   
                    cont_avScore[count,1] = np.sum(av_scores[1:,1])/av_scores[0,1] 
                    cont_bestScore[count,1] = av_scores[1:,1].max()            
                if (pd_strats == 2).any() == True:
                    cont_avScore[count,2] = np.sum(av_scores[1:,2])/av_scores[0,2]
                    cont_bestScore[count,2] = av_scores[1:,2].max() 
                if (pd_strats == 3).any() == True:
                    cont_avScore[count,3] = np.sum(av_scores[1:,3])/av_scores[0,3]
                    cont_bestScore[count,3] = av_scores[1:,3].max()
                if (pd_strats == 5).any() == True:
                    cont_avScore[count,4] = np.sum(av_scores[1:,4])/av_scores[0,4]
                    cont_bestScore[count,4] = av_scores[1:,4].max()
                av_scores = np.zeros(((epoch*9)+1, 5), dtype=np.float64)
                count += 1
        
        # If the simulation is populated entirely by a single strategy-set, halt        
        if m%epoch == 0:
            for n in range(5):
                if freq_densities[m-1,n] == np.square(matrix_dim):
                    stop = 1
                    break
            if stop == 1:
                fig = plt.figure(figsize=(4, 4))
                ax = fig.add_subplot(111)
                cax = ax.matshow(new_pop,cmap=cmap)
                plt.show()
                break    
    
    # Create epoch index to accompany frequency and score matrices
    x = np.zeros((its+1, 1))
    for i in range(its+1):
        x[i] = i
    y = np.zeros((int(its/epoch)+1, 1))
    for i in range(int(its/epoch)+1):
        y[i] = i
   
    # Add epoch index to matrices 
    frequencies = np.concatenate((x/epoch, freq_densities), axis=1)
    av_scores = np.concatenate((y, cont_avScore), axis=1)
    best_scores = np.concatenate((y, cont_bestScore), axis=1)
    
    # Output diagnostic files
    if TYPE == 'RPS':
        np.savetxt('freq.out', frequencies, fmt = '%.4f ' + '%g '*6, delimiter=' ', comments='', header = 'epoch num_rock num_pap num_sci num_NE num_CC num_GA')
        np.savetxt('av_scores.out', av_scores, fmt = '%.4f', delimiter=' ', comments='', header = 'epoch rock pap sci NE CC GA')
        np.savetxt('best_scores.out', best_scores, fmt = '%.4f', delimiter=' ', comments='', header = 'epoch rock pap sci NE CC GA')
    else:
        np.savetxt('70by70.out', frequencies, fmt = '%.4f ' + '%g '*5, delimiter=' ', comments='', header = 'epoch num_def num_coop num_TFT num_GTFT num_GA')
        np.savetxt('av_scores.out', av_scores, fmt = '%.4f', delimiter=' ', comments='', header = 'epoch defect coop TFT GTFT GA')
        np.savetxt('best_scores.out', best_scores, fmt = '%.4f', delimiter=' ', comments='', header = 'epoch defect coop TFT GTFT GA')
    np.savetxt('GA_stratList.out', GA_stratList, fmt = '%4.2f', delimiter=' ', comments='', header = '1 2 3 4 5 6 7 8')