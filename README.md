# ReinforcementLearning
 
Trying to understand how RL works

## Draft
The policy moves the entity which is in an environment

Therefore the policy takes in an entity and an entity takes in an environment
or should we take in the entity into the environment instead?

entity
    - every entity has an environment
    - move
    - get current state
    - get current reward

environment
    - show current state
    - show rewards
    - return reward for a state?
    - possible states

policy
    - calculates the value function
    - calculate the best action from value funtion
    - takes action for the entity
    - get the reward from the entity -> environment?

## Value Iteration

In order for value iteration to work, we would first need to know the transition probabilities from a state/action to another state. So what we might be able to do is to first create a Neural Network that tries to map the state transition probabilities then use that to perform value iteration on. I think for that to work, we would have to use a policy while updating the state transition mapping.

Since a value function is dependent on the state transition probabilities as well as the policy of choice, should i also create another network that tries to find the value function given the current policy and state transition mapping?

Here is an algorithm taken from **cs229**.

1. initialize pi randomly.
2. repeat {
    a. execute pi in the mdp for some number of trials. 
    b. using the accumulated experience in the MDP, update our estimates for Psa (and R, if applicable)
    c. apply the value iteration with the esteimated state transition probabilities and rewards to get a new estimated value function V. (initialize calculating the value iteration with the previously found solution so that it converges faster)
    d. Update pi to be the greedy policy with respect to V.
}

## Fitted Value Iteration

1. randomly initialize n states 
2. initialize theta = 0
3. Repeat {
    for each of the states i{
        for each action a{
            sample k next states
            q(a) = 1/k sum(R(s) + gamma*Value(s+1)) //this is the estimation of the value function
        }
        set y(i) = max(q(a)) //this is the estimation of the value function for the best policy
    }
    set theta = argmin(1/2*sum_over_n(theta*state - y)^2 //revise the theta to create the model of the value function.
}

in order to pick the best action given the value function, you would sample s' k times so that you can know which action has the highest expected value, we then pick that value. But there are other better ways of doing this.

If we have another model for the state transition, then we can know which action to take without having to sample multiple times to get the expected value. 
