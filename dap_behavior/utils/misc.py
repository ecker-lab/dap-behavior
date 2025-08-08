from datetime import datetime
import json

def load_json(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as file:
        return json.load(file)

def timestamp(mode="long"):
    if mode == "short":
        return datetime.now().strftime("%y%m%d-%H%M")
    elif mode == "short_seconds":
        return datetime.now().strftime("%y%m%d-%H%M%S")
    else:
        return datetime.now().strftime("%Y-%m-%dT%H-%M-%S%Z")

