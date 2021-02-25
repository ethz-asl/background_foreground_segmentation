from bfseg.utils.evaluation import evaluate_model_multiple_epochs_and_datasets

USER = TO_DEFINE

save_folder = f"/cluster/scratch/{USER}/evaluation"

model_to_evaluate = {
    f'/cluster/scratch/{USER}/pretrained_models/'
    'model_epoch_42_group_norm.h5':
        42
}

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
]

accuracies, mean_ious = evaluate_model_multiple_epochs_and_datasets(
    pretrained_dirs=[*model_to_evaluate.keys()],
    epochs_to_evaluate=[*model_to_evaluate.values()],
    datasets_names_to_evaluate=[d[0] for d in datasets_to_evaluate],
    datasets_scenes_to_evaluate=[d[1] for d in datasets_to_evaluate],
    save_folder=save_folder)
