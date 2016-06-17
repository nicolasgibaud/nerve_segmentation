# -*- coding: utf-8 -*-
"""Project configuration paths variables

Module to create the global variables of the project.
The configuration is described is the SETTINGS.json
which must be in the same folder as this file.
"""
import json
import os

# path to SETTINGS.json file (in current directory)
settings = os.path.dirname(__file__) +'/SETTINGS.json'

# import paths
PATHS = json.loads(open(settings).read())
# path to the main folders
ROOT = PATHS["ROOT"]
# path to the folder of data
DATA = os.path.join(ROOT, "data/")

print("Project configuration loaded")
