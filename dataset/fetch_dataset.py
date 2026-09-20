import urllib.request
import pathlib
import random

curr_dir = pathlib.Path(__file__).parent
INPUT_NAME = "input.txt"

def fetch_dataset() -> None:
    input_path = curr_dir / INPUT_NAME

    # Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
    if not input_path.exists():
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
        urllib.request.urlretrieve(names_url, input_path)

    docs = [l.strip() for l in open(input_path).read().strip().split('\n') if l.strip()] # list[str] of documents
    random.shuffle(docs)

    print(f"num docs: {len(docs)}") 

if __name__ == "__main__":
    fetch_dataset()
