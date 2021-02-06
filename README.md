# Ice Cream Gridworld

## Problem
The gridworld represents a discrete operational space with a specific relationship between spaces—what is intuitively thought of for 2D environments. For a holonomic point mass robot system operating within this operational space, we can build a 1:1 mapping to the robot configuration space, and thus use the gridworld to visualize the robot system.

The gridworld can have obstacles, that is, “forbidden” states that the robot cannot enter.  They are still states in the state space, but any transition (from any start state, taking any action) ending in an obstacle state will have probability 0.

The canonical gridworld augments this state space with a particular action space which includes either one step motion into any orthogonally adjacent cell or no motion at all (for a total of 5 possible actions).  It has associated noisy transition probabilities such that any motion action has a 1-p_e chance of succeeding as desired, but a p_e chance of error, evenly divided among the 4 alternate destinations (the three non-chosen directions or staying still).  If, after the error motion is taken into account, the destination state lies within an obstacle or off the edge of the gridworld, the result is that the robot doesn’t move.

The above description should allow you to define the system dynamics for a discrete state canonical gridworld system.  We will consider the example gridworld shown in the attachment (which we will call its map), with obstacles defined by the grey “X” squares.

## Testing Code
To simulate and manually enter inputs for the system, run `python gridworld.py`. To run value and policy iteration to generate optimal policies for various system parameters, run `python gridworld.py test`.

## Multi-Agent Gridworld
The gridworld problem is then extended to include multiple robots on the grid at the same time. Each robot can act independently, and therefore the state space is much larger, because each robot can be in any location.

Now, with multiple robots, it is a good idea to avoid collisions. Therefore, a negative reward (penalty) R_C is added to the system such that for every robot that occupies the same location as at least one other robot, R_C is added to the total reward for that state.

To test value and policy iteration for the multi-agent problem, run `python gridworld_multi_agent.py test`. Example results can be found in [two_agent_optimal_policies.txt](https://github.com/logan-hernandez/ice-cream-gridworld/blob/main/two_agent_optimal_policies.txt). To summarize, value iteration takes around 70 iterations and 12 minutes, while policy iteration takes less than 10 iterations and just a few minutes. Both methods result in the same policies, giving some verification of correctness.

Policy and value lists are in a predictable order based on the possible states. For example, for 2 agents on a 5x5 grid, the following order of states corresponds to the order of values/actions.
* Format: [(x_agent_1, y_agent_1), (x_agent_2, y_agent_2)]
* [(0,0), (0,0)]
* [(0,0), (0,1)]
* [(0,0), (0,2)]
* [(0,0), (0,3)]
* [(0,0), (0,4)]
* [(0,0), (1,0)]
* ...
* [(4,4), (4,3)]
* [(4,4), (4,4)]
