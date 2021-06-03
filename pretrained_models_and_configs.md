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
- Edit the placeholder `PRETRAINED_MODELS_FOLDER` in the `config.yml` files to be the above path:
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
|   1062   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 10:1  | - |
|   1063   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 4:1  | - |
|   1064   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay fraction 10% | `pickelhaube_segmentation_garage*` |
|   1066   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay fraction 5% | - |
|   1067   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 200:1  | - |
|   1069   |  `train_binary_segmodel_base.py`    | NYU -> Garage, finetuning | - |
|   1070   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 1:1, batch norm  | - |
|   1071   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 3:1, batch norm  | - |
|   1072   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 4:1, batch norm  | - |
|   1073   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 10:1, batch norm  | - |
|   1074   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 20:1, batch norm  | - |
|   1075   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay ratio 200:1, batch norm | - |
|   1077   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay fraction 5%, batch norm | - |
|   1078   |  `train_binary_segmodel_base.py`    | NYU -> Garage, replay fraction 10%, batch norm| - |
|   1080   |  `train_binary_segmodel_base.py`    | NYU -> Garage, finetuning, batch norm | - |
|   1112   |  `train_binary_segmodel_base.py`    | NYU -> Garage -> Construction, replay fraction 0.1, starting from exp 1064 | - |
|   1183   |  `train_binary_segmodel_base.py`    | NYU -> Construction, replay ratio 1:1  | - |
|   1184   |  `train_binary_segmodel_base.py`    | NYU -> Construction, replay ratio 3:1  | - |
|   1185   |  `train_binary_segmodel_base.py`    | NYU -> Construction, replay ratio 10:1  | - |
|   1187   |  `train_binary_segmodel_base.py`    | NYU -> Construction, replay ratio 20:1  | - |
|   1188   |  `train_binary_segmodel_base.py`    | NYU -> Construction, replay fraction 5% | - |
|   1189   |  `train_binary_segmodel_base.py`    | NYU -> Construction, replay fraction 10% | `pickelhaube_segmentation_rumlang*` |
|   1190   |  `train_binary_segmodel_base.py`    | NYU -> Construction, finetuning | - |
|   1217   |  `train_binary_segmodel_base.py`    | NYU -> Construction -> Garage, replay fraction 10%, starting from exp 1189 | `crossdomain_nyutorumlangtocla*` |
|   1219   |  `train_binary_segmodel_base.py`    | NYU -> Construction -> Garage, finetuning, starting from exp 1190 | `crossdomain_nyutorumlangtocla*` |
|   1223   |  `train_binary_segmodel_base.py`    | NYU -> Garage -> Construction, finetuning, starting from exp 1069 | `crossdomain_nyutoclatorumlang*` |
|   1286   |  `train_binary_segmodel_distillation.py`    | NYU -> Construction, output distillation with λ = 0.5 | - |
|   1287   |  `train_binary_segmodel_distillation.py`    | NYU -> Construction, output distillation with λ = 1 | - |
|   1288   |  `train_binary_segmodel_distillation.py`    | NYU -> Construction, output distillation with λ = 10 | - |
|   1289   |  `train_binary_segmodel_distillation.py`    | NYU -> Construction, output distillation with λ = 50 | - |
|   1290   |  `train_binary_segmodel_distillation.py`    | NYU -> Construction, feature distillation with λ = 0.5 | - |
|   1291   |  `train_binary_segmodel_distillation.py`    | NYU -> Construction, feature distillation with λ = 10 | - |
|   1292   |  `train_binary_segmodel_distillation.py`    | NYU -> Construction, feature distillation with λ = 50 | - |
|   1293   |  `train_binary_segmodel_distillation.py`    | NYU -> Construction, feature distillation with λ = 1 | - |
|   1312   |  `train_binary_segmodel_base.py`    | NYU -> Office, replay fraction 10% | `pickelhaube_segmentation_office*` |
|   1313   |  `train_binary_segmodel_base.py`    | NYU -> Office, finetuning          | - |
|   1314   |  `train_binary_segmodel_base.py`    | NYU -> Office -> Construction, replay fraction 10%, starting from exp 1312 | `crossdomain_nyutoofficetorumlang*` |
|   1317   |  `train_binary_segmodel_base.py`    | NYU -> Office -> Construction, finetuning, starting from exp 1313          | `crossdomain_nyutoofficetorumlang*` |
|   1318   |  `train_binary_segmodel_base.py`    | NYU -> Office -> Garage, replay fraction 10%, starting from exp 1312       | `crossdomain_nyutoofficetocla*`     |
|   1320   |  `train_binary_segmodel_base.py`    | NYU -> Office -> Garage, finetuning, starting from exp 1313                | `crossdomain_nyutoofficetocla*`     |
|   1322   |  `train_binary_segmodel_base.py`    | NYU -> Garage -> Office, replay fraction 10%, starting from exp 1064       | `crossdomain_nyutoclatooffice*`     |
|   1325   |  `train_binary_segmodel_base.py`    | NYU -> Garage -> Office, finetuning, starting from exp 1069                | `crossdomain_nyutoclatooffice*`     |
|   1326   |  `train_binary_segmodel_base.py`    | NYU -> Construction -> Office, replay fraction 10%, starting from exp 1189 | `crossdomain_nyutorumlangtooffice*` |
|   1329   |  `train_binary_segmodel_base.py`    | NYU -> Construction -> Office, finetuning, starting from exp 1190          | `crossdomain_nyutorumlangtooffice*` |
|   1340   |  `train_binary_segmodel_base.py`    | NYU -> Office, replay fraction 5% |
|   1341   |  `train_binary_segmodel_base.py`    | NYU -> Office, replay ratio 1:1     |
|   1342   |  `train_binary_segmodel_base.py`    | NYU -> Office, replay ratio 3:1     |
|   1343   |  `train_binary_segmodel_base.py`    | NYU -> Office, replay ratio 4:1     |
|   1345   |  `train_binary_segmodel_base.py`    | NYU -> Office, replay ratio 20:1    |
|   1346   |  `train_binary_segmodel_base.py`    | NYU -> Office, replay ratio 200:1   |
|   1351   |  `train_binary_segmodel_distillation.py`    | NYU -> Office, output distillation with λ = 0.5  |
|   1353   |  `train_binary_segmodel_distillation.py`    | NYU -> Office, output distillation with λ = 1    |
|   1354   |  `train_binary_segmodel_distillation.py`    | NYU -> Office, output distillation with λ = 10   |
|   1359   |  `train_binary_segmodel_distillation.py`    | NYU -> Office, feature distillation with λ = 0.5 |
|   1360   |  `train_binary_segmodel_distillation.py`    | NYU -> Office, feature distillation with λ = 1   |
|   1361   |  `train_binary_segmodel_distillation.py`    | NYU -> Office, feature distillation with λ = 10  |
|   1362   |  `train_binary_segmodel_distillation.py`    | NYU -> Office, feature distillation with λ = 50  |
|   1363   |  `train_binary_segmodel_EWC.py`    | NYU -> Office, EWC with λ = 0.5       |
|   1364   |  `train_binary_segmodel_EWC.py`    | NYU -> Office, EWC with λ = 1         |
|   1365   |  `train_binary_segmodel_EWC.py`    | NYU -> Office, EWC with λ = 50        |
|   1366   |  `train_binary_segmodel_EWC.py`    | NYU -> Office, EWC with λ = 10        |
|   1371   |  `train_binary_segmodel_EWC.py`    | NYU -> Construction, EWC with λ = 0.5 |
|   1372   |  `train_binary_segmodel_EWC.py`    | NYU -> Construction, EWC with λ = 1   |
|   1373   |  `train_binary_segmodel_EWC.py`    | NYU -> Construction, EWC with λ = 10  |
|   1374   |  `train_binary_segmodel_EWC.py`    | NYU -> Construction, EWC with λ = 50  |
|   1379   |  `train_binary_segmodel_base.py`    | NYU -> Office, replay ratio 10:1  |
|   1380   |  `train_binary_segmodel_distillation.py`    | NYU -> Office, output distillation with λ = 50   |
|   1381   |  `train_binary_segmodel_EWC.py`    | NYU -> Garage, EWC with λ = 0.5 |
|   1382   |  `train_binary_segmodel_EWC.py`    | NYU -> Garage, EWC with λ = 1   |
|   1383   |  `train_binary_segmodel_EWC.py`    | NYU -> Garage, EWC with λ = 50  |
|   1384   |  `train_binary_segmodel_EWC.py`    | NYU -> Garage, EWC with λ = 10  |
|   1393   |  `train_binary_segmodel_distillation.py`    | NYU -> Garage, output distillation with λ = 10   |
|   1394   |  `train_binary_segmodel_distillation.py`    | NYU -> Garage, output distillation with λ = 1    |
|   1395   |  `train_binary_segmodel_distillation.py`    | NYU -> Garage, output distillation with λ = 50   |
|   1396   |  `train_binary_segmodel_distillation.py`    | NYU -> Garage, output distillation with λ = 0.5  |
|   1401   |  `train_binary_segmodel_distillation.py`    | NYU -> Garage, feature distillation with λ = 0.5 |
|   1402   |  `train_binary_segmodel_distillation.py`    | NYU -> Garage, feature distillation with λ = 10  |
|   1403   |  `train_binary_segmodel_distillation.py`    | NYU -> Garage, feature distillation with λ = 1   |
|   1404   |  `train_binary_segmodel_distillation.py`    | NYU -> Garage, featture distillation with λ = 50  |
|   1410   |  `train_binary_segmodel_base.py`    | NYU -> Construction, replay ratio 4:1   |
|   1411   |  `train_binary_segmodel_base.py`    | NYU -> Construction, replay ratio 200:1 |


