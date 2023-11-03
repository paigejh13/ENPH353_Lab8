import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

        tp = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
              (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1),
              (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2),
              ]
        initialVals = [-40, -20, 0, 20, 40, 40, 20, 0, -20, -40,
                        40, 30, 20, 10, 0, -10, -20, -30, -40,
                        -40, -30, -20, -10, 0, 10, 20, 30, 40]
        i = 0

        for i in range(0,len(initialVals)):
            self.q[tp[i]] = initialVals[i]
            i += 1

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.
        file = open('Qvals', 'r')
        self.q = pickle.load(file)
        file.close

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        file = open('Qvals', 'wb')
        pickle.dump(self.q, file)
        file.close

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((int(state), action[0]), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        
        # 0 = learn, 1 = exploit
        numList = [0,1]
        weights = [self.epsilon, 1-self.epsilon]
        randomNumber = random.choices(numList, weights=weights, k=1)

        if randomNumber[0] == 0:
            choices = [self.actions[0], self.actions[1], self.actions[2]]
            weights = [1/3, 1/3, 1/3]
            action = random.choices(choices, weights=weights, k=1)
        
        else:
            LeftQ = self.getQ(state, [self.actions[0]])
            RightQ = self.getQ(state, [self.actions[1]])
            ForwardQ = self.getQ(state, [self.actions[2]])

            if LeftQ > RightQ and LeftQ >ForwardQ:
                action = [self.actions[0]]
                Q = LeftQ
            elif RightQ > LeftQ and RightQ > ForwardQ:
                action = [self.actions[1]]
                Q=RightQ
            else:
                action = [self.actions[2]]
                Q = ForwardQ
        if (return_q == True):
            return(action, Q)
        else:
            return action

        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        Q1 = self.getQ(state1, action1)
        Q2s = [self.getQ(state2, [self.actions[0]]), self.getQ(state2, [self.actions[1]]), self.getQ(state2, [self.actions[2]])]
        Q2 = max(Q2s)
        Q = self.alpha*(reward+ self.gamma*Q2 - Q1)
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        self.q[(state1,action1[0])] = Q

