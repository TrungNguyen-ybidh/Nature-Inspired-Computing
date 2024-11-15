import pandas as pd
from evo import Evo
from profiler import profile, Profiler
import contextlib
from functools import partial

@profile
def calculate_overallocation(assignments, ta_data):
    # Sum of each row to get total sections assigned per TA
    ta_assignments = assignments.sum(axis=1)
    print("TA Assignments (per TA):", ta_assignments)
    # Calculate the penalty for over-allocation based on the max assignments allowed
    penalty = sum(
        max(0, ta_assignments[i] - ta_data.iloc[i]["max_assigned"])
        for i in range(len(ta_assignments))
    )
    print("Overallocation penalty:", penalty)
    return penalty

@profile
def calculate_conflicts(assignments, section_data):
    # Group sections by their meeting times
    time_conflicts = 0

    for ta_id in assignments.index:  # Loop over each TA
        ta_schedule = assignments.loc[ta_id]  # Sections assigned to this TA

        # Find sections assigned at the same time by merging with section_data on 'daytime'
        assigned_sections = section_data[section_data.index.isin(ta_schedule[ta_schedule == 1].index)]
        if assigned_sections['daytime'].duplicated().any():  # Check for any duplicate times (conflicts)
            time_conflicts += 1  # Only count one conflict per TA

    print("Time conflicts:", time_conflicts)
    return time_conflicts

@profile
def calculate_undersupport(assignments, section_data):
    # Check number of TAs assigned per section
    ta_counts = assignments.sum(axis=0)
    print("TA Counts per Section:", ta_counts)

    # Calculate the penalty for undersupport based on the minimum TAs required
    penalty = sum(
        max(0, section_data.loc[i, 'min_ta'] - ta_counts[i])
        for i in range(len(ta_counts))
    )
    print("Undersupport penalty:", penalty)
    return penalty

@profile
def calculate_unwilling(assignments, ta_data):
    unwilling_count = 0
    for ta_id in assignments.index:
        preferences = ta_data.loc[ta_id, '0':'16']  # Preference columns
        assigned_sections = assignments.loc[ta_id] == 1  # Sections assigned to the TA
        unwilling_count += sum(preferences[assigned_sections.index[assigned_sections]] == 'U')
    return unwilling_count

@profile
def calculate_unpreferred(assignments, ta_data):
    unpreferred_count = 0
    for ta_id in assignments.index:
        preferences = ta_data.loc[ta_id, '0':'16']  # Preference columns
        assigned_sections = assignments.loc[ta_id] == 1  # Sections assigned to the TA
        unpreferred_count += sum(preferences[assigned_sections.index[assigned_sections]] == 'W')
    return unpreferred_count

@profile
def redistribution_agent(solutions, ta_data):
    new_solution = solutions[0].copy()
    ta_assignments = new_solution.sum(axis=1) - ta_data["max_assigned"]
    overallocated_tas = ta_assignments[ta_assignments > 0]

    for ta_id, excess in overallocated_tas.items():
        for section in new_solution.columns:
            if new_solution.at[ta_id, section] == 1:
                for alt_ta_id in ta_data.index:
                    if ta_assignments[alt_ta_id] <= 0:
                        new_solution.at[ta_id, section] = 0
                        new_solution.at[alt_ta_id, section] = 1
                        break
    return new_solution

@profile
def conflict_resolver_agent(solutions, section_data):
    new_solution = solutions[0].copy()
    for ta_id in new_solution.index:
        ta_schedule = new_solution.loc[ta_id]
        assigned_sections = section_data[section_data.index.isin(ta_schedule[ta_schedule == 1].index)]
        duplicate_times = assigned_sections[assigned_sections['daytime'].duplicated(keep=False)]

        for section in duplicate_times.index:
            for alt_section in section_data.index:
                if alt_section != section and section_data.loc[alt_section, 'daytime'] != section_data.loc[section, 'daytime']:
                    new_solution.at[ta_id, section] = 0
                    new_solution.at[ta_id, alt_section] = 1
                    break
    return new_solution

@profile
def support_maximizer_agent(solutions, section_data):
    new_solution = solutions[0].copy()
    ta_counts = new_solution.sum(axis=0)
    undersupported_sections = section_data["min_ta"] - ta_counts
    undersupported_sections = undersupported_sections[undersupported_sections > 0]

    for section, shortfall in undersupported_sections.items():
        for ta_id in new_solution.index:
            if new_solution.at[ta_id, section] == 0:
                new_solution.at[ta_id, section] = 1
                shortfall -= 1
                if shortfall <= 0:
                    break
    return new_solution

@profile
def preference_optimizer_agent(solutions, ta_data):
    new_solution = solutions[0].copy()
    for ta_id in new_solution.index:
        preferences = ta_data.loc[ta_id, '0':'16']
        assigned_sections = new_solution.loc[ta_id] == 1
        for section in assigned_sections.index[assigned_sections]:
            if preferences[section] in ['U', 'W']:
                for alt_section in preferences.index:
                    if str(alt_section) in new_solution.columns:
                        if preferences[alt_section] == 'P' and new_solution.at[ta_id, str(alt_section)] == 0:
                            new_solution.at[ta_id, section] = 0
                            new_solution.at[ta_id, str(alt_section)] = 1
                            break
    return new_solution

@profile
def load_data():
    # Load the assignment, TA, and section data from CSV files
    assignments = pd.read_csv('test1.csv', header=None)
    ta_data = pd.read_csv('tas.csv').set_index("ta_id")
    section_data = pd.read_csv('sections.csv').set_index("section")
    return assignments, ta_data, section_data

@profile
def main():
    # Load data
    assignments, ta_data, section_data = load_data()

    # Initialize Evo instance
    evo_instance = Evo()

    # Define fitness criteria using the penalty functions
    evo_instance.add_fitness_criteria("overallocation", lambda sol: calculate_overallocation(sol, ta_data))
    evo_instance.add_fitness_criteria("conflicts", lambda sol: calculate_conflicts(sol, section_data))
    evo_instance.add_fitness_criteria("undersupport", lambda sol: calculate_undersupport(sol, section_data))
    evo_instance.add_fitness_criteria("unwilling", lambda sol: calculate_unwilling(sol, ta_data))
    evo_instance.add_fitness_criteria("unpreferred", lambda sol: calculate_unpreferred(sol, ta_data))

    # Register agents with Evo, passing required arguments using partial
    evo_instance.add_agent("redistribution_agent", partial(redistribution_agent, ta_data=ta_data))
    evo_instance.add_agent("conflict_resolver_agent", partial(conflict_resolver_agent, section_data=section_data))
    evo_instance.add_agent("support_maximizer_agent", partial(support_maximizer_agent, section_data=section_data))
    evo_instance.add_agent("preference_optimizer_agent", partial(preference_optimizer_agent, ta_data=ta_data))

    # Add initial solution (starting point) to Evo
    evo_instance.add_solution(assignments.copy())

    # Run the evolution with a time limit of 5 minutes
    time_limit = 300 # 5 minutes in seconds
    best_solutions = evo_instance.evolve(n=1000000, dom=100, status=1000, time_limit=time_limit)

    # Process the Pareto-optimal solutions
    summary_data = []
    for idx, solution in enumerate(best_solutions):
        if isinstance(solution, tuple) and len(solution) == 2 and isinstance(solution[1], list):
            metrics = solution[1]  # Extract metrics from tuple
            if len(metrics) != 5 or not all(isinstance(val, (int, float)) for val in metrics):
                print(f"Skipping invalid solution with metrics: {metrics}")
                continue
        else:
            print(f"Skipping invalid solution format: {solution}")
            continue

        group_name = f"solution_{idx}"
        summary_data.append([group_name] + metrics)

    # Create a DataFrame from summary_data
    if summary_data:
        summary_df = pd.DataFrame(
            summary_data,
            columns=["groupname", "overallocation", "conflicts", "undersupport", "unwilling", "unpreferred"]
        )
        summary_df.to_csv("pareto_solutions.csv", index=False)
        print("Pareto-optimal solutions have been saved to pareto_solutions.csv")
    else:
        print("No valid solutions found. Skipping CSV output.")

    # Save detailed assignments and evaluation scores to a text file
    try:
        chosen_solution = summary_data[0]  # Choose the first valid solution
        with open("solution_details.txt", "w") as file:
            # Assigned sections for each TA
            file.write("Assigned Sections for Each TA:\n")
            for ta_id in assignments.index:
                assigned_sections = assignments.loc[ta_id][assignments.loc[ta_id] == 1].index.tolist()
                file.write(f"TA {ta_id}: {', '.join(map(str, assigned_sections))}\n")

            file.write("\nAssigned TAs for Each Section:\n")
            for section_id in assignments.columns:
                assigned_tas = assignments[assignments[section_id] == 1].index.tolist()
                file.write(f"Section {section_id}: {', '.join(map(str, assigned_tas))}\n")

            file.write("\nEvaluation Scores for Chosen Solution:\n")
            file.write(f"Overallocation Penalty: {chosen_solution[1]}\n")
            file.write(f"Conflicts Penalty: {chosen_solution[2]}\n")
            file.write(f"Undersupport Penalty: {chosen_solution[3]}\n")
            file.write(f"Unwilling Penalty: {chosen_solution[4]}\n")
            file.write(f"Unpreferred Penalty: {chosen_solution[5]}\n")
        print("Solution details have been saved to solution_details.txt")
    except IndexError:
        print("No valid solutions available for detailed reporting.")

    # Save the profiler report
    save_profiler_report()


def save_profiler_report():
    """Save profiler report to a file."""
    with open("profiler_report.txt", "w") as file:
        with contextlib.redirect_stdout(file):
            Profiler.report()
    print("Profiler report has been saved to profiler_report.txt")


if __name__ == "__main__":
    main()
    save_profiler_report()









