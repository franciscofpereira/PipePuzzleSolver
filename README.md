# Pipe Puzzle Solver

- This AI agent is capable of solving pipe puzzles. The objective is to connect all of the pipes in order to allow for the flow of water and avoid water leaks.

- You can find similar puzzles here: https://hexapipes.vercel.app/square/5
c
- Once the AI agent finds the solution, a window pops up with a visualizer of the solved puzzle.


# Problem Example (3x3)

![image](https://github.com/franciscofpereira/PipePuzzleSolver/assets/147160910/e91d464a-e844-480b-a25e-ddb808ee944a)


# Visualization of search (15x15)

https://github.com/franciscofpereira/PipePuzzleSolver/assets/147160910/4dad0ef9-5b88-4826-9f34-e96172361429

# Strategy Employed

- The problem was solved using a constraint propagation algorithm that prunes out the search space followed by a depth limited first search to generate the state nodes for uncertain actions.

- When selecting the next action to perform we used the Most Constrained Variable (MCV) in order to choose the variables with the least possible values.

- Refer to <solution_presentation.pdf> in the <src> directory for more details on the employed strategy.
