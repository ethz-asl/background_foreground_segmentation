"""
These Experiments are designed in a way that they only use part of the training / evaluation dataset.
E.g. only use images from the first 20s of the run
"""

from bfseg.experiments.SemSegWithDepthExperiment import SemSegWithDepthExperiment
from bfseg.experiments.SemSegExperiment import SemSegExperiment
from bfseg.data.meshdist.dataLoader import DataLoader


def getTimestampFilters(routes, startTimestamp, duration, max_count=None):
  """ Helper function that generates a "timestamp" filter object, that can be passed to the dataset loader

  Args:
    routes: List containing names of routes that are available in the dataset. e.g. [cam0,cam1] or [traj_1, traj_2]
    startTimestamp: First timestamp that should be captured
    duration: number in ms
    max_count: Max number of images in validation set (or None)

  """
  # empty filters
  train_filters = {}
  validation_filters = {}

  if max_count is not None:
    validation_filters['max_count'] = max_count

  for route in routes:
    train_filters[route] = {
        'timestamp': {
            'lower_bound': startTimestamp,
            'upper_bound': startTimestamp + duration
        }
    }

    validation_filters[route] = {
        'timestamp': {
            'lower_bound': startTimestamp + duration + 1,
            'upper_bound': float('inf')
        }
    }

  return train_filters, validation_filters


def addTimeBasedParams(parser):
  """ Helper function that adds new parameters to the argparser"""
  parser.add_argument('--dataset',
                      type=str,
                      default="CLA",
                      choices=['CLA', 'VICON'])
  parser.add_argument('--routes', nargs="+", default=None)

  parser.add_argument('--train_duration',
                      type=float,
                      default=10,
                      help="train_experiments duration in seconds [s]")

  parser.add_argument('--start_timestamp',
                      type=float,
                      default=None,
                      help="Timestamp where training should start")


def loadDefaultRoutes(dataset):
  """ Helper function that loads the dafault available routes for each dataset  """
  routes = []
  if dataset == "CLA":
    routes = ['cam0', 'cam1', 'cam2']
  elif dataset == "VICON":
    routes = ["walking_ppl", "rgb_2", "rgb_1", "no_ppl_some_dist"]
  else:
    raise ValueError("Unknown dataset " + dataset)
  return routes


def getDataLoader(config, loadDepth):
  """ Returns a dataloader object with images that are filtered based on their timestamps """
  # load routes that should be used. If none specified, use all as default
  if config.routes is None:
    routes = loadDefaultRoutes(config.dataset)
  else:
    routes = config.routes

  if config.start_timestamp is None:
    config.start_timestamp = 1582125462.069531 if config.dataset == "CLA" else 1606729654.146901

  # get filters for given config
  train_filter, valid_filter = getTimestampFilters(routes,
                                                   config.start_timestamp,
                                                   config.train_duration)
  # Get a dataloader to load training images
  return DataLoader(config.train_path, [config.image_h, config.image_w],
                    shuffle=False,
                    validationDir=config.validation_path,
                    validationMode="CLA",
                    batchSize=config.batch_size,
                    loadDepth=loadDepth,
                    trainFilter=train_filter,
                    validationFilter=valid_filter,
                    verbose=True)


class TimestampBasedSemSegWithDepthExperiment(SemSegWithDepthExperiment):
  """ Experiment that uses SemSeg + Depth but only trains on a subset of all images.  """

  def __init__(self):
    super(TimestampBasedSemSegWithDepthExperiment, self).__init__()

  def _addArguments(self, parser):
    """ add pseudo label specific arguements """
    super(TimestampBasedSemSegWithDepthExperiment, self)._addArguments(parser)
    addTimeBasedParams(parser)

  def loadDataLoader(self):
    self.dl = getDataLoader(self.config, True)


class TimestampBasedSemSegExperiment(SemSegExperiment):
  """ Helper function that loads the dafault available routes for each dataset"""

  def __init__(self):
    super(TimestampBasedSemSegExperiment, self).__init__()

  def _addArguments(self, parser):
    """ add pseudo label specific arguements """
    super(TimestampBasedSemSegExperiment, self)._addArguments(parser)
    addTimeBasedParams(parser)

  def loadDataLoader(self):
    self.dl = getDataLoader(self.config, False)
