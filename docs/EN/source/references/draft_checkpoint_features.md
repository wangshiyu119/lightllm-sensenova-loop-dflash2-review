# Target features for DFlash2

Drafters that use intermediate hidden states concatenate the execution-layer
outputs listed in `target_layer_ids` (or `dflash_config.target_layer_ids`). Integer layer
IDs refer to execution order, including repeated execution of shared Loop layers.

## Final normalized target features

A DFlash2 checkpoint trained with an additional final normalized target feature
can declare it explicitly:

```json
{
  "target_layer_ids": [3, 27, 51, 54],
  "target_feature_sources": [3, 27, 51, 54, "final_postnorm"]
}
```

The example concatenates five hidden states. With a target hidden size of 4096,
`fc.weight` has shape `[4096, 20480]`. The collector normalizes the final hidden
state with the target's final normalization weights, then appends it after the
intermediate-layer outputs in execution order. A missing final hidden state or a
feature order that does not match `target_layer_ids` raises an error.

Older exports with the same list in `loop_teacher.feature_sources` are accepted.
When both declarations are present they must agree.
