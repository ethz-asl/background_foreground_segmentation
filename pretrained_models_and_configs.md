## Download and set up the pretrained models and the configuration files.
- Download from [here](https://drive.google.com/drive/folders/106YqZGZpcWqFPGwzW5IyVJzHnd_uurgG?usp=sharing) the folder containing the pretrained models and the configuration files.
In the following, let us denote the path where the folder is saved via the environmental variable `SAVED_FOLDER`, e.g.:
    ```bash
    export SAVED_FOLDER=/home/user/configs_and_models
    ```
- Define the environmental variable `PRETRAINED_MODELS_FOLDER` to be the path of the subfolder containing the pretrained models, e.g.:
    ```bash
    export PRETRAINED_MODELS_FOLDER=${SAVED_FOLDER}/models
    ```
- Edit the placeholder `PRETRAINED_MODELS_FOLDER` in the `config.yml` files to be above path:
    ```bash
    cd ${SAVED_FOLDER}
    find . -type f -name '*.yml' -exec sed -i "s@PRETRAINED_MODELS_FOLDER@${PRETRAINED_MODELS_FOLDER}@g" {} +
    ```

## Re-run the training
To re-run a training job, use the `config.yml` from the experiment folder downloaded as detailed above, and run the corresponding training script shown in the tabel below. For instance, for experiment ID 1059, from the root folder run:
```bash
python train_binary_segmodel_base.py with ${SAVED_FOLDER}/configs/1059/config.yml
```
| Experiment ID  | Training script to use | Description | Used in Launchfile |
| ---- | ---- | ---- | ---- |
|   1059   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 20:1 | - |
|   1060   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 3:1  | - |
|   1061   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 1:1  | - |
|   1062   |  `train_binary_segmodel_base.py`    | 
|   1063   |  `train_binary_segmodel_base.py`    |
|   1064   |  `train_binary_segmodel_base.py`    |
|   1066   |  `train_binary_segmodel_base.py`    |
|   1067   |  `train_binary_segmodel_base.py`    |
|   1069   |  `train_binary_segmodel_base.py`    |
|   1070   |  `train_binary_segmodel_base.py`    |
|   1071   |  `train_binary_segmodel_base.py`    |
|   1072   |  `train_binary_segmodel_base.py`    |
|   1073   |  `train_binary_segmodel_base.py`    |
|   1074   |  `train_binary_segmodel_base.py`    |
|   1075   |  `train_binary_segmodel_base.py`    |
|   1077   |  `train_binary_segmodel_base.py`    |
|   1078   |  `train_binary_segmodel_base.py`    |
|   1080   |  `train_binary_segmodel_base.py`    |
|   1112   |  `train_binary_segmodel_base.py`    |
|   1183   |  `train_binary_segmodel_base.py`    |
|   1184   |  `train_binary_segmodel_base.py`    |
|   1185   |  `train_binary_segmodel_base.py`    |
|   1187   |  `train_binary_segmodel_base.py`    |
|   1188   |  `train_binary_segmodel_base.py`    |
|   1189   |  `train_binary_segmodel_base.py`    |
|   1190   |  `train_binary_segmodel_base.py`    |
|   1217   |  `train_binary_segmodel_base.py`    |
|   1219   |  `train_binary_segmodel_base.py`    |
|   1223   |  `train_binary_segmodel_base.py`    |
|   1286   |  `train_binary_segmodel_distillation.py`    |
|   1287   |  `train_binary_segmodel_distillation.py`    |
|   1288   |  `train_binary_segmodel_distillation.py`    |
|   1289   |  `train_binary_segmodel_distillation.py`    |
|   1290   |  `train_binary_segmodel_distillation.py`    |
|   1291   |  `train_binary_segmodel_distillation.py`    |
|   1292   |  `train_binary_segmodel_distillation.py`    |
|   1293   |  `train_binary_segmodel_distillation.py`    |
|   1312   |  `train_binary_segmodel_base.py`    |
|   1313   |  `train_binary_segmodel_base.py`    |
|   1314   |  `train_binary_segmodel_base.py`    |
|   1317   |  `train_binary_segmodel_base.py`    |
|   1318   |  `train_binary_segmodel_base.py`    |
|   1320   |  `train_binary_segmodel_base.py`    |
|   1322   |  `train_binary_segmodel_base.py`    |
|   1325   |  `train_binary_segmodel_base.py`    |
|   1326   |  `train_binary_segmodel_base.py`    |
|   1329   |  `train_binary_segmodel_base.py`    |
|   1340   |  `train_binary_segmodel_base.py`    |
|   1341   |  `train_binary_segmodel_base.py`    |
|   1342   |  `train_binary_segmodel_base.py`    |
|   1343   |  `train_binary_segmodel_base.py`    |
|   1345   |  `train_binary_segmodel_base.py`    |
|   1346   |  `train_binary_segmodel_base.py`    |
|   1351   |  `train_binary_segmodel_distillation.py`    |
|   1353   |  `train_binary_segmodel_distillation.py`    |
|   1354   |  `train_binary_segmodel_distillation.py`    |
|   1359   |  `train_binary_segmodel_distillation.py`    |
|   1360   |  `train_binary_segmodel_distillation.py`    |
|   1361   |  `train_binary_segmodel_distillation.py`    |
|   1363   |  `train_binary_segmodel_EWC.py`    |
|   1364   |  `train_binary_segmodel_EWC.py`    |
|   1365   |  `train_binary_segmodel_EWC.py`    |
|   1366   |  `train_binary_segmodel_EWC.py`    |
|   1371   |  `train_binary_segmodel_EWC.py`    |
|   1372   |  `train_binary_segmodel_EWC.py`    |
|   1373   |  `train_binary_segmodel_EWC.py`    |
|   1374   |  `train_binary_segmodel_EWC.py`    |
|   1379   |  `train_binary_segmodel_base.py`    |
|   1380   |  `train_binary_segmodel_distillation.py`    |
|   1381   |  `train_binary_segmodel_EWC.py`    |
|   1382   |  `train_binary_segmodel_EWC.py`    |
|   1383   |  `train_binary_segmodel_EWC.py`    |
|   1384   |  `train_binary_segmodel_EWC.py`    |
|   1393   |  `train_binary_segmodel_distillation.py`    |
|   1394   |  `train_binary_segmodel_distillation.py`    |
|   1395   |  `train_binary_segmodel_distillation.py`    |
|   1401   |  `train_binary_segmodel_distillation.py`    |
|   1402   |  `train_binary_segmodel_distillation.py`    |
|   1403   |  `train_binary_segmodel_distillation.py`    |
|   1404   |  `train_binary_segmodel_distillation.py`    |
|   1410   |  `train_binary_segmodel_base.py`    |
|   1411   |  `train_binary_segmodel_base.py`    |


