import os

from bfseg.utils.evaluation import evaluate_model_multiple_epochs_and_datasets

USER = TO_DEFINE

save_folder = f"/cluster/scratch/{USER}/evaluation"
pretrained_models_folder = f'/cluster/scratch/{USER}/pretrained_models/'

id_and_epoch = [(1064, 100), (1069, 100)]

datasets_to_evaluate = [
    # NYU.
    ("NyuDepthV2Labeled", None),
    ("NyuDepthV2Labeled", "kitchen"),
    ("NyuDepthV2Labeled", "bedroom"),
    # Hive labels.
    ("BfsegValidationLabeled", "CLA"),
    ("OfficeRumlangValidationLabeled", "OFFICE"),
    ("OfficeRumlangValidationLabeled", "RUMLANG"),
    # Pseudo-labels.
    ("BfsegCLAMeshdistLabels", None),
    ("MeshdistPseudolabels", "garage1"),
    ("MeshdistPseudolabels", "garage2"),
    ("MeshdistPseudolabels", "garage3"),
    ("MeshdistPseudolabels", "rumlang_full"),
    ("MeshdistPseudolabels", "rumlang2"),
    ("MeshdistPseudolabels", "rumlang3"),
    ("MeshdistPseudolabels", "office4"),
    ("MeshdistPseudolabels", "office5"),
    ("MeshdistPseudolabels", "office4_2302"),
    ("MeshdistPseudolabels", "office4_2402"),
    ("MeshdistPseudolabels", "office6_2502")
]

accuracies = {}
mean_ious = {}

for id_, epoch_ in id_and_epoch:
  id_folder = os.path.join(save_folder, str(id_))
  if (not os.path.isdir(id_folder)):
    os.makedirs(id_folder)
  if (not id_ in accuracies):
    accuracies[id_] = {}
    mean_ious[id_] = {}

  model_to_evaluate = os.path.join(pretrained_models_folder,
                                   f'{id_}_model_epoch_{epoch_}.h5')

  accuracies[id_][epoch_], mean_ious[id_][
      epoch_] = evaluate_model_multiple_epochs_and_datasets(
          pretrained_dirs=model_to_evaluate,
          epochs_to_evaluate=epoch_,
          datasets_names_to_evaluate=[d[0] for d in datasets_to_evaluate],
          datasets_scenes_to_evaluate=[d[1] for d in datasets_to_evaluate],
          save_folder=id_folder)
