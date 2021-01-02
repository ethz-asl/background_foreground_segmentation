from bfseg.experiments.SemSegExperiment import SemSegExperiment
from bfseg.utils.losses import comdined_pseudo_label_loss


class SemSegWithPseudoLabelsExperiment(SemSegExperiment):

  def __init__(self):
    super(SemSegWithPseudoLabelsExperiment, self).__init__()

  def _addArguments(self, parser):
    """ add pseudo label specific arguements """
    super(SemSegWithPseudoLabelsExperiment, self)._addArguments(parser)
    parser.add_argument('--pseudo_label_threshold',
                        type=float,
                        help='Threshold for pseudo labels confidence',
                        default=0.9)
    parser.add_argument(
        '--pseudo_label_weight',
        type=float,
        help='Relative weight between pseudo labels and meshdist label',
        default=0.1)

  def getLoss(self):
    """ change loss function to train_experiments with pseudo labels"""
    return comdined_pseudo_label_loss(self.config.pseudo_label_weight,
                                      self.config.pseudo_label_weight,
                                      self.config.loss_balanced)
