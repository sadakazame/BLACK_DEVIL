# target_script.py (合格バージョン)
def greet(name: str) -> None:
    # Blackに直させるために、あえて少し詰めて書く
    message = "Hello " + name
    print(message)


# 正しい型（文字列）で呼び出す
greet("Gemini")
