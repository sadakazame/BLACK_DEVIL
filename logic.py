# logic.py
def format_name(first: str, last: str) -> str:
    # 苗字を大文字にして結合する
    return f"{first} {last.upper()}"

if __name__ == "__main__":
    print(format_name("tarou", "yamada"))