# -*- coding: utf-8 -*-
"""Project configuration paths variables

Module to create the global variables of the project.
The configuration is described is the SETTINGS.json
which must be in the same folder as this file.
"""
import json
import os

# path to SETTINGS.json file (in current directory)
current_dir_path = os.path.dirname(os.path.realpath(__file__))
settings = os.path.join(current_dir_path, 'SETTINGS.json')
settings = settings.replace("\\", "/")

# import paths
PATHS = json.loads(open(settings).read())
# path to the main folders
ROOT_DIR = PATHS["ROOT_DIR"]
# path to the folder of data
DATA_DIR = os.path.join(ROOT_DIR, "data")
