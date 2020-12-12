#!/usr/bin/python3

# 1. randomly initialize n states
# 2. initialize theta = 0
# 3. Repeat {
#    for each of the states i{
#        for each action a{
#            sample k next states
#            q(a) = 1/k sum(R(s) + gamma*Value(s+1)) //this is the estimation of the value function
#        }
#        set y(i) = max(q(a)) //this is the estimation of the value function for the best policy
#    }
#    set theta = argmin(1/2*sum_over_n(theta*state - y)^2 //revise the theta to create the model of the value function.
# }

class FittedValueIteration:
    def __init__(self):
        pass
