from _test_format import run_list_of_tests

# IMPORT ALL TESTS HERE
from test_helper_functions import test_get_nodes_x_hops_away, test_get_numb_of_nb_x_hops_away
from test_main_flow import test_readme_code

# ADD ALL TESTS TO THIS LIST
tests = [
    test_get_nodes_x_hops_away,
    test_get_numb_of_nb_x_hops_away, 
    test_readme_code
]

run_list_of_tests(tests)
