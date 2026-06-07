# main_app.py
from logic import multiply

def run_app():
    val1, val2 = 10, 5
    result = multiply(val1, val2)
    print(f"RESULT:{result}")

if __name__ == "__main__":
    run_app()