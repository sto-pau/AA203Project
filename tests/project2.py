import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import time

class TMatrix:
    def __init__(self, total_a, total_s, total_sp):
        self.depth = total_a
        self.rows = total_s
        self.columns = total_sp
        self.T = []
        for action in range(total_a):
            self.T.append( csr_matrix((self.rows , self.columns)) )
    def update(self, a, s, sp, val):
        self.T[a-1][s-1, sp-1] = val
    def indx(self, a, s, sp):
        return self.T[a-1][s-1, sp-1]

class RMatrix:
    def __init__(self, total_a, total_s):
        self.depth = total_a
        self.rows = total_s
        self.R = []
        for action in range(total_a):
            self.R.append( csr_matrix((self.rows , 1)) )
    def update(self, a, s, val):
        self.R[a-1][s-1, 0] = val
    def indx(self, a, s):
        return self.R[a-1][s-1, 0]

def createTnR(df):

    T = TMatrix(total_a, total_s, total_sp)
    R = RMatrix(total_a, total_s)

    denominators = df.groupby(['s', 'a']).size().reset_index(name='count')
    numerators = df.groupby(['s', 'a', 'sp']).size().reset_index(name='count')

    total_a = df['a'].max()
    total_s = df['s'].max()
    total_sp = df['sp'].max()

    #Transitions
    for combo in range(numerators.shape[0]):
        #select row, grab the counts
        s = numerators.iloc[combo]['s']
        a = numerators.iloc[combo]['a']
        sp = numerators.iloc[combo]['sp']
        val = numerators.iloc[combo]['count'] / denominators.loc[(denominators['s'] == s) & (denominators['a'] == a)]['count']
        T.update(a,s,sp, val)
    #reward
    for combo in range(numerators.shape[0]):
        #rho = sum for states of all actions
        s = numerators.iloc[combo]['s']
        a = numerators.iloc[combo]['a']
        #filter data frame to just include the combination we want
        filtered_cases = df[(df['s'] == s) & (df['a'] == a)]
        #add all the rewards
        rho = filtered_cases['r'].sum() / denominators.loc[(denominators['s'] == s) & (denominators['a'] == a)]['count']
        R.update(a, s, rho)

    return T, R

def readCSV(path):
    #from this data we obtain the dataset and the variables
    df = pd.read_csv(path)     
    return df

def write_policy(policy, filename):
    with open(filename, 'w') as f:
        for state in range(len(policy)):
            f.write("{}\n".format(policy[state]+1))

def kernel(Q_known, Q_unknown):
    '''
    s is a vector of all known states
    sp is the state we calculating its smoothed value
    dist is a matrix
    rows correspond to individual sp
    columns to distance to each s
    '''
    e = 10**-100 #hyperparameter to prevent divide by 0

    '''
    inverse of manhattan distance between states
    create a matrix of known state value
    and then to subtract the unknown sp from each state
    gives matrixes with a row for every sp
    with its difference to all states
    compute for pos and vel
    summation of differences is L1
    '''
    #create a weight for each known state for each unknown state
    return 1 / np.maximum(( (abs( (np.tile(Q_known[:,1], (len(Q_unknown[:,1]), 1))) - Q_unknown[:, 1, np.newaxis] )) + 
                           (abs( (np.tile(Q_known[:,2], (len(Q_unknown[:,2]), 1))) - Q_unknown[:, 2, np.newaxis] ))) , e) #l1 distance #equation 8.5

def smooth(Q_known, Q_unknown):
    '''
    Q_s is a matrix of all known states and values
    sp is a vector of states

    states are all rows first column of s
    dist is matrix of rows corresponding to all s
    column is for each sp

    return the weighted addition
    of the normalized distance to all known states
    and the known actions 
    to find the smoothed results for the unknown states 
    '''
    dist = kernel(Q_known, Q_unknown)
    return np.dot(dist / np.sum(dist,axis=1).reshape(-1,1), Q_known[:,-1]) #equation 8.3 + 8.4

def plotXYXY(x1,y1,x2,y2):

    #xy plot for debugging showing known & unknown states by pos & vel
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(x1, y1)
    ax2.scatter(x2, y2)
    ax1.set_xlim(0, 500)
    ax2.set_xlim(0, 500)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    plt.show()

def heatmap(x, y, heat):
    #visualize the known Q values after Qtrain before smoothing
    # Create a hexbin so data values don't block each other
    plt.hexbin(x, y, C=heat)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Heat Map')
    plt.show()

def Qtrain(Q, df, gamma, alpha, residual, loops):

    #iterate through all given data updating Q(s,a)    
    for loop_through_data in range(loops):
        U = np.max(Q, axis = 1)
        for row in df.itertuples(): #options are iterrows(), values()
            #index for all values
            s = row.s - 1
            a = row.a - 1
            sp = row.sp - 1
            r = row.r - 1
            Q[s,a] += alpha * (r + gamma * np.max(Q[sp]) - Q[s,a])
        U_new = np.max(Q, axis = 1)
        #if all new improvements are no greater than the residual
        #break early
        if np.all(abs((U - U_new)) <  residual):
            print("broke on loop: ", loop_through_data)
            break

    print("max residual: ", np.max( abs( (U - U_new) ) ),3)
    #smooth over Q values of 0
    #done for each action seperately

    del U, U_new, s, a, r, sp, df
    gc.collect()

    return Q

def KernelSmoothing(total_a, Q):

    for action in range(total_a):

        #peel off action
        Q_a = np.array([Q[:,action]]).T #visualizing as column vector
    
        #add index 
        states = np.array([np.arange(1, total_s+1)]).T #visualizing as column vector, start at 1

        print("Q_shape", np.shape(Q_a))
        
        Q_pos = (states - 1) % 500
        Q_vel = (states - 1) // 500

        Q_zip = np.hstack(( np.hstack(( np.hstack(( states, Q_pos)) , Q_vel)) , Q_a)) 

        #mask to remove just the Q=0 values (or could use where)
        #create *boolean mask* where the Q_a column is equal to 0
        mask = (Q_zip[:,-1] == 0) #all rows, check the action column
        #use the bool mask to index the array
        Q_unknown = Q_zip[mask,:]
        Q_known = Q_zip[~mask,:]

        ##for visualizing what states are known and the known Qs
        #plotXYXY(Q_known[:,1],Q_known[:,2],Q_unknown[:,1],Q_unknown[:,2])
        #heatmap(Q_pos.flatten(), Q_vel.flatten(), Q_a.flatten()) #x, y, heat

        #clean up memory
        del Q_a, states, Q_zip, mask, Q_pos, Q_vel
        gc.collect()

        #smooth the action value function for unknown states
        Q_unknown[:,-1] = smooth(Q_known, Q_unknown)

        #combine to reform Q and peel off state
        Q_smoothed = np.concatenate((Q_known, Q_unknown))
        #update the action row in Q
        Q_smoothed = Q_smoothed[Q_smoothed[:,0].argsort()]
        Q[:,action] = Q_smoothed[:,-1]

        ##for visualizing updated action value states
        #heatmap(Q_smoothed[:,1], Q_smoothed[:,2], Q_smoothed[:,-1]) #x, y, heat

        #clean up memory
        del Q_smoothed, Q_known, Q_unknown
        gc.collect()

if __name__ == '__main__':
    
    start = time.time()    
    
    #Read in data into a data frame
    #data should be in the same folder, input to function should be target file name and return file name
    save_path = 'C:/Users/stocc/OneDrive/Documents/Winter2023/AA228/ProjectRepo/project2/data/'
    file_name = 'large'    
    path = save_path + file_name + '.csv'
    
    df = readCSV(path) #obtain the dataset and the variables from csv file  
    
    #size of actions and states needs to be hardcoded 
    #as no garuntee max action is in exploration 
    problem_dict = {
        'small': (4, 100, 0.95),
        'medium': (7, 50000, 1),
        'large': (9, 312020, 1)       
    }

    total_a, total_s, gamma = problem_dict[file_name]
    total_sp = total_s

    #initialize Q(s,a) matrix
    Q = np.zeros((total_s, total_a))

    #iterate through data, updating each line
    '''hyper paramters, gamma from problem statement
    alpha to be tuned, larger is faster convergence but
    potentially suboptimal'''
    alpha = 10**-2
    residual = 0.1 #Bellman Residual, [10^-3, 0.1]
    loops = 10

    Q = Qtrain(Q, df, gamma, alpha, residual, loops)

    if file_name == 'medium':
        KernelSmoothing(total_a, Q)

    #pick the optimal policy with greedy Value Function Policy
    #return the best action for each state (row)
    pi = np.argmax(Q, axis = 1)        

    filename = save_path + file_name + '.policy'
    write_policy(pi, filename)

    #total run time 
    end = time.time()
    print(end - start) 
    print("here")