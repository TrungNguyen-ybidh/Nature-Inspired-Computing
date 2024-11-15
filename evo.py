import random as rnd
import copy
from functools import reduce
import numpy as np
import time
import csv

class Evo:
    def __init__(self):
        self.pop = {}     # evaluation --> solution
        self.fitness = {} # name --> objective function
        self.agents = {}  # name --> (operator function, num_solutions_input)

    def add_fitness_criteria(self, name, f):
        """ Register an objective with the environment """
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        """ Register an agent with the environment
        The operator (op) defines how the agent tweaks a solution.
        k defines the number of solutions input to the agent. """
        self.agents[name] = (op, k)

    def add_solution(self, sol):
        """ Add a solution to the population """
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[eval] = sol

    def get_random_solutions(self, k=1):
        """ Pick k random solutions from the population """
        if len(self.pop) == 0:  # No solutions in the population
            return []
        else:
            solutions = tuple(self.pop.values())
            return [copy.deepcopy(rnd.choice(solutions)) for _ in range(k)]

    def run_agent(self, name):
        """ Invoke a named agent on the population """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)

    def dominates(self, p, q):
        """ Check if solution p dominates solution q """
        pscores = np.array([score for name, score in p])
        qscores = np.array([score for name, score in q])
        score_diffs = qscores - pscores
        return min(score_diffs) >= 0 and max(score_diffs) > 0.0

    def reduce_nds(self, S, p):
        """ Helper to remove dominated solutions """
        return S - {q for q in S if self.dominates(p, q)}

    def get_non_dominated_solutions(self):
        """Return a dictionary of all non-dominated solutions in the population."""
        non_dominated_solutions = set(self.pop.keys())  # Start with all solutions as non-dominated

        # Compare each solution with every other solution in the population
        for p in self.pop.keys():
            for q in self.pop.keys():
                if p != q and self.dominates(q, p):  # If q dominates p, remove p from non-dominated solutions
                    non_dominated_solutions.discard(p)
                    break  # No need to check further; we know p is dominated

        # Return the non-dominated solutions with their solution data
        return {solution: self.pop[solution] for solution in non_dominated_solutions}

    def remove_dominated(self):
        """ Remove dominated solutions from the population """
        nds = reduce(self.reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k: self.pop[k] for k in nds}

    def save_non_dominated_to_csv(self, filename):
        """ Save non-dominated solutions to a CSV file in the required format """
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header as specified
            headers = ['groupname', 'overallocation', 'conflicts', 'undersupport', 'unwilling', 'unpreferred']
            writer.writerow(headers)

            # Write each solution's data
            for i, (solution_id, solution_data) in enumerate(self.pop.items()):
                # Generate a group name based on solution index, e.g., "Group1", "Group2", etc.
                groupname = f"Group{i + 1}"

                # Extract the required fields in the specified order
                row = [
                    groupname,
                    solution_data.get('overallocation', 0),  # Default to 0 if key not present
                    solution_data.get('conflicts', 0),
                    solution_data.get('undersupport', 0),
                    solution_data.get('unwilling', 0),
                    solution_data.get('unpreferred', 0)
                ]

                writer.writerow(row)

    def evolve(self, n=1, dom=100, status=1000, time_limit=300):
        """
        Run random agents with a time limit
        n: Number of agent invocations
        dom: How frequently to remove dominated solutions
        status: How frequently to output the current population
        time_limit: Maximum time allowed for this function (in seconds)
        """
        agent_names = list(self.agents.keys())
        start_time = time.time()  # Record the start time

        for i in range(n):
            # Check if time limit is exceeded
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"Time limit of {time_limit} seconds reached.")
                break

            # Run a random agent
            pick = rnd.choice(agent_names)
            print(f"Iteration {i}: Running agent '{pick}'")
            self.run_agent(pick)

            # Remove dominated solutions periodically
            if i % dom == 0:
                self.remove_dominated()

            # Print status periodically
            if i % status == 0:
                print("Iteration:", i)
                print("Population size:", len(self.pop))
                print(self)

        # Final cleanup
        self.remove_dominated()

        # Log final state
        print(f"Evolution completed. Total iterations: {i}")
        print(f"Final population size: {len(self.pop)}")
        print("Non-dominated solutions:")
        print(self.get_non_dominated_solutions())

        # Ensure population is not empty
        if not self.pop:
            print("Warning: Population is empty. No solutions found.")
            return []

        # Return non-dominated solutions
        return self.get_non_dominated_solutions()

    def __str__(self):
        """ Output the solutions in the population """
        result = ""
        for eval, sol in self.pop.items():
            result += str(eval) + ":\t" + str(sol) + "\n"
        return result

