# cayley-py_1112121706

# PilgrimAttnRes Training Analysis

## Project topic

**Investigation of the training process of the PilgrimAttnRes model architecture.**

The main goal is to find an efficient training strategy that minimizes training time while achieving a **solution rate (%Solv) above 90%** in beam search.

The project focuses on understanding how different training schedules and learning-rate transitions affect the model's learning dynamics and the final beam-search performance.

---

## Objectives

### 1. Analysis of the results from the original paper

Analyze the results table from the project paper.

The original Kaggle implementation is available here:

[cayleypy-rw-modelbaselines-megaminx](https://www.kaggle.com/code/alexandervc/cayleypy-rw-modelbaselines-megaminx?utm_source=chatgpt.com)

### 2. Reduction of training time

Investigate whether model training time can be reduced while maintaining a solution rate of **%Solv > 90%**.

The analysis focuses particularly on the behavior of individual model layers during training and on the effect of different learning-rate schedules.

---

# Training Code Architecture

The model training and experiment-management code is organized into four main classes:

* `Curve`
* `CheckpointInfo`
* `Trainer` / `TrainerLight`
* `ExperimentManager`

## `Curve`

`Curve` stores the history and configuration of a particular training run.

It contains:

* training configuration specific to the current curve;
* training and validation metrics;
* layer and network metrics;
* checkpoint information;
* training schedule;
* information required to continue training.

In this sense, `Curve` can be viewed as an **extension of the general training configuration (`cfg`) for a particular training run**.

The class also automatically generates a descriptive name for the training curve based on its learning-rate schedule. This name is then used when creating files containing the curve, checkpoints, and analysis results.

The class provides utilities for:

* saving and loading curves;
* cloning curves;
* extending a curve to continue training;
* building data structures for plotting;
* saving training metrics to CSV;
* working with checkpoints and training history.

## `CheckpointInfo`

`CheckpointInfo` stores information about a particular model checkpoint.

If `Curve` can be considered an extension of the training configuration, `CheckpointInfo` can be considered an **extension of the information stored in `paper_dict` for a particular trained model**.

It stores information such as:

* checkpoint file;
* epoch;
* training loss;
* validation loss;
* moving-average validation loss;
* whether the checkpoint is the best checkpoint;
* training time.

## `Trainer`

`Trainer` contains the actual model-training procedures.

Its responsibilities include:

* generating training/validation data;
* training one epoch;
* validating one epoch;
* applying the learning-rate schedule;
* calculating training and validation losses;
* saving checkpoints;
* determining when a checkpoint should be saved.

`TrainerLight` is a lightweight version used when additional training statistics are not required. It avoids collecting computationally expensive metrics and unnecessary checkpoints, making it suitable for large numbers of exploratory training runs.

The calculation of the additional training-dynamics metrics is deliberately kept outside the trainer.

## `ExperimentManager`

`ExperimentManager` prepares all components required for training a particular `Curve`.

It:

1. initializes the `PilgrimAttnRes` model;
2. initializes the Adam optimizer;
3. creates the appropriate `Trainer`;
4. loads an existing checkpoint if required;
5. moves the model and optimizer to the appropriate device;
6. starts the training process.

If `is_light=True`, the manager uses the lightweight training procedure.

The typical workflow is therefore:

```python
curve = Curve(
    schedule=schedule,
    num_epochs=num_epochs,
    cfg=cfg,
    ...
)

manager = ExperimentManager(
    curve,
    graph,
    PilgrimAttnRes,
    GROUPING_RULES,
)

manager.train(curve)
```

The architecture is designed so that only **one model needs to remain in GPU memory at a time**. Multiple `Curve` objects can be kept in memory because they contain relatively little data. Curves and models/checkpoints can therefore be loaded and processed sequentially.

---

# Additional Training-Dynamics Metrics

Initially, the main metrics used for evaluating training were validation loss and its moving average (`MA50`).

However, the same validation loss can lead to different beam-search results. Therefore, additional metrics were introduced to investigate the state and dynamics of the model during training.

## Network-level metrics

`net_metrics` describe the behavior of the model as a whole.

### `epoch_displacement`

The norm of the change in the complete parameter vector during one training epoch.

It measures **how far the model moved through parameter space during the epoch**.

### `relative_parameter_displacement`

The epoch displacement normalized by the norm of the previous parameter vector.

It measures the size of the parameter update relative to the current scale of the model parameters.

### `cosine_update`

The cosine similarity between consecutive parameter updates.

It indicates whether successive updates tend to move in similar directions or change direction.

### `update_variance`

The variance of recent epoch displacement magnitudes.

It provides an indication of how stable the magnitude of parameter updates is during training.

### `validation_improvement`

The change in validation loss between consecutive validation measurements.

Positive values indicate an improvement in validation loss; negative values indicate deterioration.

Because training and validation data are regenerated for each epoch, this metric can contain considerable stochastic variation.

### `improvement_per_displacement`

Validation improvement normalized by parameter displacement.

It measures how much validation-loss improvement is obtained per unit of parameter movement.

---

# Layer-level Metrics

`layer_metrics` describe the behavior of individual parts of the model.

A "layer" can refer to:

* the complete architecture;
* a group of layers;
* a complete `ResidualBlock`;
* an individual layer inside a residual block;
* another user-defined group of parameters.

The main layer-level metrics are:

### `grad_norm`

The norm of the gradient for the selected layer or group.

It indicates the strength of the learning signal received by the layer.

### `weight_norm`

The norm of the layer's parameters.

It indicates the scale of the parameters.

### `grad_weight_ratio`

The gradient norm relative to the weight norm.

This provides a normalized measure of how strongly the current gradient acts on the layer relative to its parameter scale.

### `adam_update_norm`

The norm of the actual parameter update produced by Adam.

This is particularly useful because the gradient magnitude alone does not determine how much the parameters change.

### `update_weight_ratio`

The Adam update magnitude relative to the weight magnitude.

This indicates the relative size of the parameter change for a particular layer.

These layer-level metrics are useful for identifying situations where some layers continue learning while others have already entered a regime of very small updates, or where a large learning rate causes parameter growth without corresponding improvement in the final result.

---

# Experiments

The experiments investigate several training strategies:

* constant learning rate;
* piecewise learning-rate schedules;
* different starting learning rates;
* different epochs for learning-rate transitions;
* different learning rates for the `input_layer` and the remaining layers;
* the behavior of individual layers during training.

A particular focus is placed on identifying the point at which a learning-rate reduction becomes beneficial.

The experiments show that the learning process is not determined by validation loss alone. Models with similar validation losses can occupy different regions of parameter space and produce different beam-search results.

This suggests that the **training trajectory and the timing of learning-rate transitions are important factors in the final model quality**.

---

# Main Findings

The experiments indicate that achieving a stable **%Solv > 90%** generally requires sufficiently long training, with approximately **500 or more epochs** being necessary for the configurations investigated.

Successful training is associated with reaching an appropriate validation-loss range while keeping the learning rate within a suitable range.

An appropriately selected **variable learning-rate schedule** can provide better results than a constant learning rate.

## Learning-rate transition

There appears to be an optimal period for switching to a lower learning rate.

If the learning rate is reduced **too early**, the model is still under-trained and does not sufficiently explore the parameter space.

If the learning rate is reduced **too late**, training time is wasted and the model can spend too long in an undesirable high-update regime. In some cases, prolonged training with an excessively large learning rate can even degrade the subsequent beam-search performance.

Therefore, the learning-rate transition should occur at an appropriate point in the training trajectory:

**too early → under-training**

**optimal transition → effective refinement**

**too late → wasted training and potentially degraded beam-search performance**

The experiments therefore suggest that there is an **optimal training curve / learning-rate schedule** for a given architecture and task.

The goal is not simply to minimize validation loss, but to find a training trajectory that reaches a good region of parameter space efficiently and produces strong beam-search performance.

---

# Notebooks

Examples and tests of the implementation are available in the following notebooks:

1. `cayleypy-rw-modelbaselines-megaminx.ipynb`
2. `cayleypy_code_tests_examples.ipynb`
3. `cayleypy_plot_table_examples.ipynb`

The notebooks contain examples of:

* model training;
* checkpoint loading;
* training-curve analysis;
* layer-level metric analysis;
* plotting;
* comparison of different learning-rate schedules;
* beam-search evaluation.

---

# Summary

This project extends the original model-training code with a structured training framework and additional tools for analyzing training dynamics.

The main research question is:

> **How can the learning-rate schedule and training process be optimized to minimize training time while maintaining a beam-search solution rate above 90%?**

The analysis suggests that the answer depends not only on the final validation loss, but also on **when and how the model moves through parameter space during training**, how individual layers respond to the learning rate, and when learning-rate transitions are made.
