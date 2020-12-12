## Inverted Pendulum Problem

## Goal

1. Try to create an optimal policy to control the inverted pendulum.

## Todo

- [x] Create physics simulation
- [ ] Think of a training strategy (reward function, game restart, etc.)
- [ ] Action state discretization
- [ ] Transition Probability Model (Dynamic Model)
- [ ] Fitted Value Iteration (Create a model for the value function)

## Draft

- In order to get the best next action. We need to know what values will the state which the action will bring us to is.
- Therefore, we create a value function model when we have a continous state space.
- But even if we know the value function, how do we choose the action which would bring about the state which has the best value?
- We can create a state transition model so that we can approximate where a certain action will bring us. 
- But we still wouldn't know which action to take... Is there a way to find the action which maximizes the value function? (what if we assume that the actions can't change very rapidly. We would only simulate actions which are similar to the current action and then greedily pick the best action out of those actions. This would save computational time because we would only have to simulate a few possible actions)
- or we can discretize the action space.
- If we already have the transition probabilities, why can't we just pick an action that would make the current state close to the target state, without even using the value function? In simple cases, we might not even need the value function, but in some cases, it's not clear or easy to identify the next best state to be in (unlike many control problems) thats the reason why a value function exists.
