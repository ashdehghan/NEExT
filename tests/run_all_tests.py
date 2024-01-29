from _test_format import run_list_of_tests

# IMPORT ALL TESTS HERE
from test_helper_functions import test_get_nodes_x_hops_away, test_get_numb_of_nb_x_hops_away

# ADD ALL TESTS TO THIS LIST
tests = [
    test_get_nodes_x_hops_away,
    test_get_numb_of_nb_x_hops_away
]

run_list_of_tests(tests)
