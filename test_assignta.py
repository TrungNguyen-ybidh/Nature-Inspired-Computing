import pytest
import pandas as pd
from assignta import (
    calculate_overallocation,
    calculate_conflicts,
    calculate_undersupport,
    calculate_unwilling,
    calculate_unpreferred,
)
from profiler import profile, Profiler

# Fixture to load data once and provide it to each test function
@pytest.fixture
def load_data():
    # Load CSV files
    assignments = pd.read_csv('test3.csv', header=None)
    ta_data = pd.read_csv('tas.csv').set_index("ta_id")  # Ensure TA IDs are indexed for easy lookup
    section_data = pd.read_csv('sections.csv').set_index("section")  # Ensure sections are indexed
    return assignments, ta_data, section_data

# Fixture to set up the Evo instance with sample data
# Test for calculate_overallocation
def test_calculate_overallocation(load_data):
    assignments, ta_data, section_data = load_data
    expected_penalty = 23
    profiled_calculate_overallocation = profile(calculate_overallocation)
    assert profiled_calculate_overallocation(assignments, ta_data) == expected_penalty

# Test for conflicts penalty
def test_calculate_conflicts(load_data):
    assignments, ta_data, section_data = load_data
    expected_conflicts = 2
    profiled_calculate_conflicts = profile(calculate_conflicts)
    assert profiled_calculate_conflicts(assignments, section_data) == expected_conflicts

# Test for undersupport penalty
def test_calculate_undersupport(load_data):
    assignments, ta_data, section_data = load_data
    expected_undersupport_penalty = 7
    profiled_calculate_undersupport = profile(calculate_undersupport)
    assert profiled_calculate_undersupport(assignments, section_data) == expected_undersupport_penalty

# Test for unwilling assignments penalty
def test_calculate_unwilling(load_data):
    assignments, ta_data, section_data = load_data
    expected_unwilling_penalty = 43
    profiled_calculate_unwilling = profile(calculate_unwilling)
    assert profiled_calculate_unwilling(assignments, ta_data) == expected_unwilling_penalty

# Test for unpreferred assignments penalty
def test_calculate_unpreferred(load_data):
    assignments, ta_data, section_data = load_data
    expected_unpreferred_penalty = 10
    profiled_calculate_unpreferred = profile(calculate_unpreferred)
    assert profiled_calculate_unpreferred(assignments, ta_data) == expected_unpreferred_penalty










