import json

def openjson(path):
    with open(path, "r") as f:
        dict = json.loads(f.read())
    return dict