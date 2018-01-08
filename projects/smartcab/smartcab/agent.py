import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pygame
import pandas as pd
import numpy as np
import os

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 
    # valid_actions = [None, 'forward', 'left', 'right']
    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5, decay_rate = -0.025, Qini = 0):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.initialQ = Qini
        self.decay_rate = decay_rate
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = 0

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.trials = -1
        self.max_trials = 100
        self.x_trials = range(0, self.max_trials)
        self.y_trials = range(0, self.max_trials)

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        self.epsilon_decay() 
        self.trials = self.trials + 1        
        if testing == True:
            self.epsilon = 0
            self.alpha = 0


        return None

    def epsilon_decay(self):
        if self.decay_rate > 0:
            self.epsilon = self.epsilon - self.decay_rate
        else:
            self.epsilon = self.epsilon*np.exp(self.decay_rate)
              
        
    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent 
        # the state consists of "waypoint" + "light" +"oncoming traffinc" + "left traffic"

        #if inputs['right'] != "forward":
        #    inputs['right'] = "not forward"
        #if inputs['left'] == "right":
        #    inputs['left'] = None
        #print "input format {}".format(inputs)
        state = tuple([waypoint] + inputs.values()[0:2]+inputs.values()[3:4])
        
        #state = tuple([waypoint] + inputs.values())
        print "state format {}".format(state)

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        q = [self.Q[state][action] for action in self.valid_actions]
        maxQ = max(q)
        count = q.count(maxQ)
        if count == 1:
            action = self.valid_actions[q.index(maxQ)]
        else:
            ind = [i for i in range(len(self.valid_actions)) if q[i] == maxQ]
            action = self.valid_actions[random.choice(ind)]

        

        return maxQ, action 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if state not in self.Q.keys():
            self.Q[state]={}
            for i in self.valid_actions:
                self.Q[state][i] = self.initialQ
            print self.Q[state]

            

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        if not self.learning:
            action = random.choice(self.valid_actions)
        
        else:
            if random.random() < self.epsilon:
                action = random.choice(self.valid_actions)
            else:
                maxQ, action = self.get_maxQ(state)
                
                
                
 
        return action


    def learn(self, state, next_state, reward, action):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        #ind = self.valid_actions.index(action)
        if state != None:
            max_Q, ind2 = self.get_maxQ(next_state)
            max_Q = 0
            reward = reward + max_Q
            self.Q[state][action] = \
            self.Q[state][action]*(1-self.alpha) + reward*self.alpha
        else:
            print "first state"
                  
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward

#        self.learn(self.previous_state, state, self.previous_reward, self.previous_action) # Q-learn
        self.learn(state, state, reward, action) # Q-learn

        self.previous_state = state
        self.previous_reward = reward
        self.previous_action = action
        print self.epsilon

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose = True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning = True, alpha = 0.4, decay_rate = -0.0375)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay = 0.001, log_metrics = True, display= False, optimized = True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test = 10, tolerance=0.05)

def multiple_run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """
    alpha_all = [float(i+5)/10 for i in xrange(3)]
    #decay_rate_all = [-0.1, -0.075, -0.05, -0.025]
    decay_rate_all = [0.02, 0.01]
    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    start = 0
    learning = True
    optimization = True
    
    for alpha in alpha_all:
        for decay_rate in decay_rate_all:
            env = Environment(verbose = True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
            agent = env.create_agent(LearningAgent, learning = learning, alpha = alpha, decay_rate = decay_rate)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
            env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
            sim = Simulator(env, update_delay = 0.001, log_metrics = True, display= False, optimized = True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
            sim.run(n_test = 10, tolerance=0.05)
            if learning == True and optimization == True:
                log_filename = os.path.join("logs", "sim_improved-learning.csv")
                logall_filename = os.path.join("logs", "sim_improved-learning_all.csv")
                new_data = pd.read_csv(log_filename)
                new_data['decay'] = decay_rate
                new_data['alpha'] = alpha
                
                if start == 0:
                    new_data.to_csv(logall_filename, index = False)
                else:
                    alldata = pd.concat([pd.read_csv(logall_filename), new_data],ignore_index=True)
                    alldata.to_csv(logall_filename, index = False)
            start+=1

if __name__ == '__main__':
    #multiple_run()
    run()
