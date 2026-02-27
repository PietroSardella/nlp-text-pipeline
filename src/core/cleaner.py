import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zà-ú\s]", "", text)
    return text