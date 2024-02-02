from typing import List, Callable


def test_wrapper(func):
    print(f"Running test {func.__name__}...")
    try:
        res = func()
        if res is False:
            print(f"Test {func.__name__} failed.")
            return False
        
        # if res is True or None
        print(f"Test {func.__name__} passed.")
        return True

    except Exception as e:
        print(e)
        return False


def run_list_of_tests(functions: List[Callable]):
    results = [test_wrapper(function) for function in functions]
    print("--------------------------------------------------")
    print(f"Passed {sum(results)} out of {len(results)} tests.")
