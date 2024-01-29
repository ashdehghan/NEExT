# How to run tests
1. run `python run_all_tests.py` from the `tests` directory.

# How to add tests
1. make a new file called `test_<name>.py` in the `tests` directory
2. add a functions that test the functionality of your code, with the following conventions
* The function takes no arguments
* The function does NOT catch unintentional exceptions
* The function returns a boolean value: this can indicate whether the output is correct, or whether the test ran without errors.
3. In the `run_all_tests.py` file, import the tests and add the test functions to the `tests` list.

