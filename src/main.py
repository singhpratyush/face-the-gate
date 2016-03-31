import sys
from getopt import getopt, GetoptError

from src.utils import refresh_data, start_gate_keeper

usage_text = './main.py [-r] [-c <camera_id>]'

if __name__ == "__main__":

    # Constants
    camera_id = 0

    options = ()
    # Process options
    try:
        options, args = getopt(
            sys.argv[1:],
            "rc:",
            [
                '--refresh-data',
                '--camera_id='
            ]
        )
    except GetoptError as err:
        print('Invalid arguments')

    for opt, arg in options:
        if opt in ('-r', '--refresh-data'):
            refresh_data()
        if opt in ('-c', '--camera-id'):
            camera_id = int(arg)

    start_gate_keeper(camera_id=camera_id)