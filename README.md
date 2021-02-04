# Ice Cream Gridworld

## Problem
The gridworld represents a discrete operational space with a specific relationship between spaces—what is intuitively thought of for 2D environments. For a holonomic point mass robot system operating within this operational space, we can build a 1:1 mapping to the robot configuration space, and thus use the gridworld to visualize the robot system.

The gridworld can have obstacles, that is, “forbidden” states that the robot cannot enter.  They are still states in the state space, but any transition (from any start state, taking any action) ending in an obstacle state will have probability 0.

The canonical gridworld augments this state space with a particular action space which includes either one step motion into any orthogonally adjacent cell or no motion at all (for a total of 5 possible actions).  It has associated noisy transition probabilities such that any motion action has a 1-p_e chance of succeeding as desired, but a p_e chance of error, evenly divided among the 4 alternate destinations (the three non-chosen directions or staying still).  If, after the error motion is taken into account, the destination state lies within an obstacle or off the edge of the gridworld, the result is that the robot doesn’t move.

The above description should allow you to define the system dynamics for a discrete state canonical gridworld system.  We will consider the example gridworld shown in the attachment (which we will call its map), with obstacles defined by the grey “X” squares.
