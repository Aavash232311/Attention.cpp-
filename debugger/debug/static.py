RED = "\033[91m"
RESET = "\033[0m"
GREEN = "\033[92m"

def print_check(label, passed):
    color = GREEN if passed else RED
    status = "PASS" if passed else "FAIL"
    print(f"{color}[{status}]{RESET} {label}")
