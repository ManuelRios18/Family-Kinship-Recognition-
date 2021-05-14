import json


def save_json(file_path, data):
    with open(file_path, 'w') as fp:
        json.dump(data, fp)
