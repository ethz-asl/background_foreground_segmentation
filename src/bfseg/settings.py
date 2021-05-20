from os import environ

# Set the following parameters to save the experiment logs to a MongoDB server
# through Sacred (cf. `src/bfseg/settings.py`).
EXPERIMENT_DB_HOST = None
EXPERIMENT_DB_USER = None
EXPERIMENT_DB_PWD = None
EXPERIMENT_DB_NAME = None

# Output folder of the experiments.
try:
  TMPDIR = environ['TMPDIR']
except KeyError:
  raise KeyError(
      "Please define the folder where to store the experiment outputs, by "
      "setting its path to the environmental variable 'TMPDIR', e.g., "
      "'export TMPDIR=/home/user/output_folder'.")