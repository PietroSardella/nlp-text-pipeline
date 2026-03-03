import os


def load_documents(folder_path: str) -> list[str]:
    """
    Carrega todos os arquivos .txt de uma pasta
    e retorna cada linha como um documento separado.
    """

    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                lines = file.read().splitlines()
                documents.extend(
                    [line.strip() for line in lines if line.strip()]
                )

    return documents