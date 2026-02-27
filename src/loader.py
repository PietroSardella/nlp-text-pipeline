def load_documents(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]