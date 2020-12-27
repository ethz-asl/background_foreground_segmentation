from bfseg.experiments.SemSegWithDepthExperiment import SemSegWithDepthExperiment
from bfseg.experiments.SemSegExperiment import SemSegExperiment
from bfseg.utils.losses import combined_loss
from bfseg.data.meshdist.dataLoader import DataLoader


def getTimestampFilters(routes, startTimestamp, duration, max_count = None):
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
        parser.add_argument('--dataset', type=str, default ="CLA", choices = ['CLA', 'VICON'])
        parser.add_argument('--routes', nargs ="+", default = None)
        parser.add_argument('--train_duration', type=float, default = 10, help = "train duration in seconds [s]")

def loadDefaultRoutes(dataset):
    routes = []
    if dataset == "CLA":
        routes = ['cam0', 'cam1', 'cam2']
    elif dataset == "VICON":
        routes = ["walking_ppl", "rgb_2", "rgb_1", "no_ppl_some_dist"]
    else:
        raise ValueError("Unknown dataset " + dataset)
    return routes


class TimestampBasedSemSegWithDepthExperiment(SemSegWithDepthExperiment):
    def __init__(self):
      super(SemSegWithPseudoLabelsExperiment, self).__init__()

    def _addArguments(self, parser):
        """ add pseudo label specific arguements """
        super(SemSegWithDepthExperiment, self)._addArguments(parser)
        addTimeBasedParams(parser)

    def loadDataLoader(self):

        if self.config.routes is None:
            routes = loadDefaultRoutes(self.config.dataset)
        else:
            routes = self.config.routes

        startTimestamp = 1582125462.069531 if self.config.dataset == "CLA" else 0
        train_filter, valid_filter = getTimestampFilters(routes, startTimestamp, self.config.train_duration)

        # Get a dataloader to load training images
        self.dl = DataLoader(self.config.train_path,
                             [self.config.image_h, self.config.image_w],
                             validationDir=self.config.validation_path,
                             validationMode="CLA",
                             batchSize=self.config.batch_size,
                             loadDepth=True,
                             trainFilter=train_filter,
                             validationFilter=valid_filter,
                             cropOptions={
                                 'top': 0,
                                 'bottom': 0
                             })


class TimestampBasedSemSegExperiment(SemSegExperiment):
    def __init__(self):
        super(TimestampBasedSemSegExperiment, self).__init__()

    def loadDataLoader(self):
        if self.config.routes is None:
            routes = loadDefaultRoutes(self.config.dataset)
        else:
            routes = self.config.routes

        startTimestamp = 1582125462.069531 if self.config.dataset == "CLA" else 0
        train_filter, valid_filter = getTimestampFilters(routes, startTimestamp,  self.config.train_duration)

        # Get a dataloader to load training images
        self.dl = DataLoader(self.config.train_path,
                             [self.config.image_h, self.config.image_w],
                             validationDir=self.config.validation_path,
                             validationMode="CLA",
                             batchSize=self.config.batch_size,
                             loadDepth=False,
                             trainFilter=train_filter,
                             validationFilter=valid_filter)

    def _addArguments(self, parser):
        """ add pseudo label specific arguements """
        super(TimestampBasedSemSegExperiment, self)._addArguments(parser)
        addTimeBasedParams(parser)