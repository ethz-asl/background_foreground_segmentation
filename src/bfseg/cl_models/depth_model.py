import tensorflow as tf
from tensorflow import keras
import warnings

from tensorflow.python.ops.gen_math_ops import mean

from bfseg.cl_models import BaseCLModel
from bfseg.utils.models import create_model


class DepthModel(BaseCLModel):
  r"""Model with distillation loss. Two types of distillation loss are possible:
  1. Feature distillation on the intermediate feature space (at the output of
     the encoder);
  2. Distillation on the network output.
  Args:
    run (sacred.run.Run): Object identifying the current sacred run.
    root_output_dir (str): Path to the folder that will contain the experiment
      logs and the saved models.
  """

  def __init__(self, run, root_output_dir):
    super(DepthModel, self).__init__(run=run, root_output_dir=root_output_dir)
    # Loss weights
    self.semseg_weight = self.run.config['depth_params']['semseg_weight']
    self.depth_weight = self.run.config['depth_params']['depth_weight']
    self.consistency_weight = self.run.config['depth_params']['consistency_weight']
    self.mse_weight = self.run.config['depth_params']['mse_weight']

  def _build_model(self):
    r"""Builds the models.
    """
    cl_framework = self.run.config['cl_params']['cl_framework']
    assert (cl_framework in [
        "distillation", "ewc", "finetune"
    ]), "Currently, only distillation, EWC and fine-tuning are supported."
    # NOTE: by default the model is created as trainable. The encoder can be
    # optionally be set as non-trainable through the config file.
    # CL frameworks that require a fixed, non-trainable network from which to
    # distill the information (e.g., in distillation experiments) should create
    # additional models by overloading this method and calling
    # `super()._build_model()` in the overload.
    pretrained_dir = self.run.config['cl_params']['pretrained_dir']
    should_freeze_encoder = self.run.config['network_params']['freeze_encoder']
    if (should_freeze_encoder):
      if (cl_framework not in ["distillation", "finetune"]):
        raise ValueError(
            "Freezing the encoder is only done for finetuning. If you are "
            "really sure that you want to do so for other CL frameworks, "
            "manually remove this check.")
      if (pretrained_dir is None):
        raise ValueError(
            "It is necessary to specify pretrained weights to be loaded if the "
            "encoder is to be fixed.")
      warnings.warn(
          "The encoder of the created model is set to be non-trainable.")
    self.encoder, self.model = create_model(
        model_name=self.run.config['network_params']['architecture'],
        freeze_encoder=should_freeze_encoder,
        freeze_whole_model=False,
        normalization_type=self.run.config['network_params']
        ['normalization_type'],
        **self.run.config['network_params']['model_params'])
    self.new_model = keras.Model(
        inputs=self.model.input,
        outputs=[self.encoder.output, self.model.output])
    # Optionally load the model weights.
    if (pretrained_dir is not None):
      print(f"Loading pre-trained weights from {pretrained_dir}.")
      self.new_model.load_weights(pretrained_dir)

  def _build_loss_and_metric(self):
    r"""Adds loss criteria and metrics.
    """
    self.loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
    self.loss_mae = keras.metrics.MeanAbsoluteError(name='mae_loss', dtype=tf.float32) # NEW
    self.accuracy_tracker = keras.metrics.Accuracy('accuracy', dtype=tf.float32)
    self.miou_tracker = keras.metrics.MeanIoU(num_classes=2, name='mean_iou')
    self.mae_tracker = keras.metrics.MeanAbsoluteError(name='depth_mae', dtype=tf.float32) # NEW
    # This stores optional logs about the test metrics, which is not
    # automatically handled by Keras.
    self.logs_test = {}
    # By default, no auxiliary losses are expected to be tracked.
    self._tracked_auxiliary_losses = None
    # Check if balanced cross-entropy loss should be used.
    if (self.run.config['training_params']['use_balanced_loss']):
      assert (self.run.config['cl_params']['cl_framework'] == "finetune"
             ), "Currently balanced loss is only supported with finetuning."
      self._use_balanced_loss = True
    else:
      self._use_balanced_loss = False

  def log_metrics(self, metric_type, logs, step):
    r"""Logs to sacred the metrics for the given dataset type ("train",
    "train_no_replay", "val", or "test") at the given step.
    Args:
      metric_type (str): Either "train", "train_no_replay", "val", "test", or
        the name of a dataset on which the model was evaluated: type of the
        dataset the metrics refer to.
      logs (dict): Dictionary containing the metrics to log, indexed by the
        metric name, with their value at the given step.
      step (int): Training epoch to which the metrics are referred.
    Returns:
      None.
    """
    for metric_name, metric_value in logs.items():
      self.run.log_scalar(f'{metric_type}_{metric_name}',
                          metric_value,
                          step=step)
    self.loss_tracker.reset_states()
    self.accuracy_tracker.reset_states()
    self.miou_tracker.reset_states()
    self.mae_tracker.reset_states()
    if (self._tracked_auxiliary_losses is not None):
      for aux_loss_name in self._tracked_auxiliary_losses:
        getattr(self, f"{aux_loss_name}_tracker").reset_states()

  def forward_pass(self, training, x, y, mask):
    r"""Forward pass.
    Args:
      training (bool): Whether or not the model should be in training mode.
      x (tf.Tensor): Input to the network.
      y (tf.Tensor): Ground-truth labels corresponding to the given input.
      mask (tf.Tensor): Mask for the input to consider.
    Return:
      pred_y (tf.Tensor): Network prediction.
      pred_y_masked (tf.Tensor): Masked network prediction.
      y_masked (tf.Tensor): Masked ground-truth labels (i.e., with labels only
        from the selected samples).
      loss (tf.Tensor): Loss from performing the forward pass.
    """
    # New, split up 
    y_seg = y['seg_label']
    y_depth = y['depth_label']
    mask_seg = mask['seg_mask']
    mask_depth = mask['depth_mask']

    #[_, pred_y] = self.new_model(x, training=training)
    something, predictions = self.new_model(x, training=training)
    # print(something.shape)
    [pred_y_seg, pred_y_combined, pred_y_depth] = predictions 
  
    # Semseg
    pred_y_seg_masked = tf.boolean_mask(pred_y_seg, mask_seg)
    y_seg_masked = tf.boolean_mask(y_seg, mask_seg)
    # print("Semseg loss inputs: (last 2)")
    # print("Mask seg: {}".format(mask_seg))
    # print("y_seg: {}".format(y_seg))
    # print("pred_y_seg: {}".format(pred_y_seg.shape))
    # print("pred_y_seg_masked: {}".format(pred_y_seg_masked))
    # print("y_seg_masked: {}".format(y_seg_masked))
    if (self._use_balanced_loss):
      #TODO(fmilano): Make flexible to number of classes.
      sample_weight = get_balanced_weights(labels=y_seg_masked, num_classes=2)
    else:
      sample_weight = None

    loss_semseg = self.loss_ce(y_seg_masked, pred_y_seg_masked, sample_weight=sample_weight)
    # moved inside
    pred_y_seg = tf.math.argmax(pred_y_seg, axis=-1) 
    pred_y_seg_masked = tf.boolean_mask(pred_y_seg, mask_seg)
    
    # Depth
    # print("Depth loss inputs:")
    # print("pred_y_depth: {}".format(pred_y_depth))
    # print("y_depth: {}".format(y_depth))
    
    loss_depth = ignorant_depth_loss(y_depth, pred_y_depth, theta=self.mse_weight) # remove hardcoded version
    
    loss_mse = mean_squared_error(y_depth, pred_y_depth)
    # pred_y_depth_ignorant = tf.where(tf.math.is_nan(y_depth),
    #                                tf.zeros_like(y_depth), pred_y_depth)
    # y_depth_ignorant = tf.where(tf.math.is_nan(y_depth),
    #                      tf.zeros_like(y_depth), y_depth)
    # loss_depth = self.loss_mae(y_depth_ignorant, pred_y_depth_ignorant)



    # Masking for MAE only
    pred_y_depth_masked = tf.boolean_mask(pred_y_depth, mask_depth)
    y_depth_masked = tf.boolean_mask(y_depth, mask_depth)

    # Consistency (https://github.com/ethz-asl/background_foreground_segmentation/blob/a817e09d6578427b01cac4d0f106b166caf8b402/src/bfseg/utils/losses.py#L47)
    semantic_classes = 2 # TODO: remove hardcoded version
    pred_y_seg2 = tf.expand_dims(pred_y_seg, -1)
    # print("Consistency loss inputs:")
    # print("pred_y_seg2: {}".format(pred_y_seg2))
    # print("pred_y_depth: {}".format(pred_y_depth))
    loss_consistency = sum([smooth_consistency_loss(pred_y_depth, pred_y_seg2, c) for c in range(semantic_classes)])

    # Final combined loss
    loss_combined = self.semseg_weight * loss_semseg + self.depth_weight * loss_depth + self.consistency_weight * loss_consistency 

    # Return loss dict
    loss = {'loss': loss_combined, 'loss_semseg': loss_semseg, 'loss_depth': loss_depth, 'loss_consistency': loss_consistency, 'loss_mse': loss_mse}

    # return complete loss, but only segementation predictions
    return pred_y_seg, pred_y_seg_masked, y_seg_masked, pred_y_depth, pred_y_depth_masked, y_depth_masked, loss

  def train_step(self, data):
    r"""Performs one training step with the input batch. Overrides `train_step`
    called internally by the `fit` method of the `tf.keras.Model`.
    Args:
      data (tuple): Input batch. It is expected to be of the form
        (train_x, train_y, train_mask), where:
        - train_x (tf.Tensor) is the input sample batch.
        - train_y (tf.Tensor) are the ground-truth labels associated to the
            input sample batch.
        - train_mask (tf.Tensor) is a boolean mask for each pixel in the input
            samples. Pixels with `True` mask are considered for the computation
            of the loss.
    
    Returns:
      Dictionary of metrics.
    """
    train_x, train_y, train_mask = data
    with tf.GradientTape() as tape:
      pred_y, pred_y_masked, train_y_masked, pred_y_depth, pred_y_depth_masked, train_y_depth_masked, loss = self.forward_pass(
          training=True, x=train_x, y=train_y, mask=train_mask)
    # print("Forward pass outputs: ")
    # print("Pred_y: {}".format(pred_y.shape))
    # print("Pred_y_masked: {}".format(pred_y_masked.shape))
    # print("train_y_masked: {}".format(train_y_masked.shape))
    # print("Pred_y_depth: {}".format(pred_y_depth.shape))
    # print("Pred_y_depth_masked: {}".format(pred_y_depth_masked.shape))
    # print("train_y_depth_masked: {}".format(train_y_depth_masked.shape))


    total_loss, auxiliary_losses = self._handle_multiple_losses(loss)
    # print("WEIGHTS: ")
    # print("Trainable: {}".format(len(self.new_model.trainable_weights)))
    # print(self.new_model.trainable_weights)
    # print("WEIGHTS: ")
    # print("Non-Trainable: {}".format(len(self.new_model.non_trainable_weights)))
    # print(self.new_model.non_trainable_weights)

    grads = tape.gradient(total_loss, self.new_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_weights))
    # moved inside forward pass
    #pred_y = tf.math.argmax(pred_y, axis=-1) 
    #pred_y_masked = tf.boolean_mask(pred_y, train_mask)

    # Update accuracy and loss. # TODO: fix MAE
    self.accuracy_tracker.update_state(train_y_masked, pred_y_masked)
    self.loss_tracker.update_state(total_loss)
    self.miou_tracker.update_state(train_y_masked, pred_y_masked)
    self.mae_tracker.update_state(train_y_depth_masked, pred_y_depth_masked)
    if (auxiliary_losses is not None):
      for aux_loss_name, aux_loss in auxiliary_losses.items():
        getattr(self, f"{aux_loss_name}_tracker").update_state(aux_loss)

    self.current_batch += 1
    self.performed_test_evaluation = False

    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    r"""Performs one evaluation (test/validation) step with the input batch.
    Overrides `test_step` called internally by the `evaluate` method of the
    `tf.keras.Model`.
    Args:
      data (tuple): Input batch. It is expected to be of the form
        (test_x, test_y, test_mask), where:
        - test_x (tf.Tensor) is the input sample batch.
        - test_y (tf.Tensor) are the ground-truth labels associated to the input
            sample batch.
        - test_mask (tf.Tensor) is a boolean mask for each pixel in the input
            samples. Pixels with `True` mask are considered for the computation
            of the loss.
    
    Returns:
      Dictionary of metrics.
    """
    assert (self.evaluation_type in ["test", "val"])
    test_x, test_y, test_mask = data
    pred_y, pred_y_masked, test_y_masked, pred_y_depth, pred_y_depth_masked, test_y_depth_masked, loss = self.forward_pass(
        training=False, x=test_x, y=test_y, mask=test_mask)
    # Moved inside forward pass
    #pred_y = keras.backend.argmax(pred_y, axis=-1)
    #pred_y_masked = tf.boolean_mask(pred_y, test_mask)

    total_loss, auxiliary_losses = self._handle_multiple_losses(loss)

    # Update val/test metrics.
    self.loss_tracker.update_state(total_loss)
    self.accuracy_tracker.update_state(test_y_masked, pred_y_masked)
    self.miou_tracker.update_state(test_y_masked, pred_y_masked)
    self.mae_tracker.update_state(test_y_depth_masked, pred_y_depth_masked)
    if (auxiliary_losses is not None):
      for aux_loss_name, aux_loss in auxiliary_losses.items():
        getattr(self, f"{aux_loss_name}_tracker").update_state(aux_loss)

    return {m.name: m.result() for m in self.metrics}


def smooth_consistency_loss(depth_pred, y_pred_semantic, class_number=0):
  """
    Makes sure the semantic and depth prediction match. e.g. there are not too many edges inside a
    segmentation mask.
    Taken from this paper, using similar naming convention.
    Semantics-Guided Disparity Smoothness (8):
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Scene_Understanding_Unsupervised_Monocular_Depth_Estimation_With_Semantic-Aware_Representation_CVPR_2019_paper.pdf
  """
  # print("Smooth consistency loss: ")
  # print("depth_pred: {}".format(depth_pred))
  # print("y_pred_semantic: {}".format(y_pred_semantic))

  phi = tf.cast(tf.math.equal(y_pred_semantic, class_number), dtype=tf.float32)
  phi_x = tf.roll(phi, 1, axis=-3)
  phi_y = tf.roll(phi, 1, axis=-2)

  depth_x = tf.roll(depth_pred, 1, axis=-3)
  depth_y = tf.roll(depth_pred, 1, axis=-2)
  diffx = tf.multiply(tf.math.abs(depth_pred - depth_x), 1 - tf.math.abs(
      (phi - phi_x)))
  diffx_no_nan = tf.where(tf.math.is_nan(diffx), tf.zeros_like(diffx), diffx)

  diffy = tf.multiply(tf.math.abs(depth_pred - depth_y), 1 - tf.math.abs(
      (phi - phi_y)))
  diffy_no_nan = tf.where(tf.math.is_nan(diffy), tf.zeros_like(diffy), diffy)

  return tf.keras.backend.mean(diffx_no_nan + diffy_no_nan)


def ignorant_depth_loss(depth_label, y_pred_depth, maxDepthVal=1000.0 / 10.0, theta=0.1): 
  """
  wrapper to mask all "NaN" values in depth
  """
  y_pred_depth_ignorant = tf.where(tf.math.is_nan(depth_label),
                                   tf.zeros_like(depth_label), y_pred_depth)
  depth_label = tf.where(tf.math.is_nan(depth_label),
                         tf.zeros_like(depth_label), depth_label)
  # print("Ignorant depth loss: ")
  # print("depth_label: {}".format(depth_label))
  # print("y_pred_depth_ignorant: {}".format(y_pred_depth_ignorant))

  return depth_loss_function(depth_label, y_pred_depth_ignorant, maxDepthVal=maxDepthVal, theta=theta)


def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0): 
  """ Loss Function from DenseDepth paper.
    Code taken from here https://github.com/ialhashim/DenseDepth/blob/master/loss.py
  """

  # Point-wise depth
  l_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true),
                                  axis=-1)
  # Edges
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  l_edges = tf.keras.backend.mean(tf.keras.backend.abs(dy_pred - dy_true) +
                                  tf.keras.backend.abs(dx_pred - dx_true),
                                  axis=-1)

  # Structural similarity (SSIM) index
  l_ssim = tf.keras.backend.clip(
      (1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

  # Weights
  w1 = 1.0
  w2 = 1.0
  w3 = theta

  return (w1 * l_ssim) + (w2 * tf.keras.backend.mean(l_edges)) + (
      w3 * tf.keras.backend.mean(l_depth))

def mean_squared_error(y_true, y_pred):
  y_pred_ignorant = tf.where(tf.math.is_nan(y_true),
                                   tf.zeros_like(y_true), y_pred)
  y_true_ignorant = tf.where(tf.math.is_nan(y_true),
                         tf.zeros_like(y_true), y_true)
  l_mse = tf.keras.backend.mean(tf.math.squared_difference(y_pred_ignorant, y_true_ignorant))
  return tf.keras.backend.mean(l_mse)