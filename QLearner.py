""" 			  		 			     			  	   		   	  			  	
Template for implementing QLearner  (c) 2015 Tucker Balch 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Copyright 2018, Georgia Institute of Technology (Georgia Tech) 			  		 			     			  	   		   	  			  	
Atlanta, Georgia 30332 			  		 			     			  	   		   	  			  	
All Rights Reserved 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Template code for CS 4646/7646 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Georgia Tech asserts copyright ownership of this template and all derivative 			  		 			     			  	   		   	  			  	
works, including solutions to the projects assigned in this course. Students 			  		 			     			  	   		   	  			  	
and other users of this template code are advised not to share it with others 			  		 			     			  	   		   	  			  	
or to make it available on publicly viewable websites including repositories 			  		 			     			  	   		   	  			  	
such as github and gitlab.  This copyright statement should not be removed 			  		 			     			  	   		   	  			  	
or edited. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
We do grant permission to share solutions privately with non-students such 			  		 			     			  	   		   	  			  	
as potential employers. However, sharing with other current or future 			  		 			     			  	   		   	  			  	
students of CS 7646 is prohibited and subject to being investigated as a 			  		 			     			  	   		   	  			  	
GT honor code violation. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
-----do not edit anything above this line--- 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  				  		 			     			  	   		   	  			  	
""" 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
import numpy as np 			  		 			     			  	   		   	  			  	
import random as rand 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
class QLearner(object): 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False): 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
        self.verbose = verbose 			  		 			     			  	   		   	  			  	
        self.num_actions = num_actions 			  		 			     			  	   		   	  			  	
        self.s = 0 			  		 			     			  	   		   	  			  	
        self.a = 0 	
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.max_actions = num_actions - 1
        
        self.T = self.initT(num_states, num_actions)
        self.Q = self.initWithZeros(num_states, num_actions)
        self.R = self.initWithZeros(num_states, num_actions)
        self.Tc = self.initT(num_states, num_actions)

    
    def author(self):
        return 'agizatulina3'
    
    # methods for inititalizing states
    def initT(self, num_s, num_a):
       return np.zeros((num_s, num_a, num_s))
   
    def initWithZeros(self, num_s, num_a):
       return np.zeros([num_s, num_a])
	     			  	  		   	  			  	
    def querysetstate(self, s): 			  		 			     			  	   		   	  			  	
        """ 			  		 			     			  	   		   	  			  	
        @summary: Update the state without updating the Q-table 			  		 			     			  	   		   	  			  	
        @param s: The new state 			  		 			     			  	   		   	  			  	
        @returns: The selected action 			  		 			     			  	   		   	  			  	
        """ 			  		 			     			  	   		   	  			  	
        self.s = s 	
        rar = self.rar
        
        rand_num = rand.random()
        
        
        
        action = np.argmax(self.Q[s, :])
        if rand_num < rar: action = rand.randint(0, self.max_actions)
            
        self.rar = rar * self.radr
        self.a = action
		  		 			     			  	   		   	  			  		  		 			     			  	   		   	  			  	
        if self.verbose: print "s =", s,"a =",action 			  		 			     			  	   		   	  			  	
        return action 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def query(self, s_prime, r): 			  		 			     			  	   		   	  			  	
        """ 			  		 			     			  	   		   	  			  	
        @summary: Update the Q table and return an action 			  		 			     			  	   		   	  			  	
        @param s_prime: The new state 			  		 			     			  	   		   	  			  	
        @param r: The ne state 			  		 			     			  	   		   	  			  	
        @returns: The selected action 			  		 			     			  	   		   	  			  	
        """ 		
        
        # Get variables
        s = self.s
        a = self.a
        num_states = self.num_states

        dyna = self.dyna
        rar = self.rar
        

        qMax = max(self.Q[s_prime,])
        alphaDiff = 1 - self.alpha
    
        # Update policy
        self.Q[s, a] = alphaDiff * self.Q[s, a] + self.alpha * (r + self.gamma * qMax)	 
    	
        rand_num = rand.random()
    
        if rand_num <= rar: action = rand.randint(0, self.max_actions)
        else: action = np.argmax(self.Q[s_prime,])
        
            
  	   	# Implement Dyna	
        if dyna > 0:
            self.Tc[s, a, s_prime] = self.Tc[s, a, s_prime] + 1

            self.T = self.Tc / np.sum(self.Tc[s, a, :])
            self.R[s, a] = alphaDiff * self.R[s, a] + self.alpha * r
 
            for i in range(dyna):
                # Random state and action
                s_dn = rand.randint(0, num_states -1)
     
                a_dn = rand.randint(0, self.max_actions)
                
                sDyna_prime = self.T[s_dn, a_dn, :].argmax()
                
                rD = self.R[s_dn, a_dn]
                q = self.Q[s_dn, a_dn]
                qMax = max(self.Q[sDyna_prime,])
                
                # Update Q hallucin
                self.Q[s_dn, a_dn] = alphaDiff * q + self.alpha * ( rD + self.gamma * qMax)
     
                
        # Set variables
        self.s = s_prime
        self.rar = rar * self.radr
        self.a = action	 

 	   		   	  			  	
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r 
			  		 			     			  	   		   	  			  	
        return action 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
if __name__=="__main__": 			  		 			     			  	   		   	  			  	
    print '' 			  		 			     			  	   		   	  			  	
