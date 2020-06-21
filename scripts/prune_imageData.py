import json
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='Remove imageData field from all .json files in tree.')
parser.add_argument('path',
    type=str,
    help="Path to parent of all target JSON files."
)


def json_delete_field(data_path, field):
    """Delete a top level `field` from JSON file at `data_path`."""
    data_path = Path(data_path)
    assert data_path.exists()
    assert data_path.suffix == '.json', 'Only JSON files supported'

    print(data_path)

    with open(data_path, 'r') as data_file:
        data = json.load(data_file)
        print(data.keys())

    if field in data.keys():
        print(f'Found {field} in {data_path}. Deleting.')
        data.pop(field, None)

    with open(data_path, 'w') as data_file:
        data = json.dump(data, data_file, indent=4)


def main():
    args = parser.parse_args()

    json_paths = list(Path(args.path).rglob('*.json'))

    if len(list(json_paths)) == 0:
        print(f'No json files found below {args.path}. Exiting.')

    else:
        for data_path in json_paths:
            json_delete_field(data_path, 'imageData')


if __name__ == '__main__':
    main()
