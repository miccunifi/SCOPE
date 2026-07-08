# Conditional family/severity pre-editor

Files:

- `conditional_attention_preeditor.py`
  - FiLM-conditioned U-Net.
  - Inputs: image, normalized severity, codec family id.
  - Outputs: same as your old model.

- `conditioning.py`
  - Codec family mapping.
  - Severity normalization to `[0, 1]`.
  - Batch condition tensor preparation.

- `train_conditional_patch.py`
  - Blocks to paste into your current trainer:
    - argparse args
    - model construction
    - condition preparation
    - model call replacement
    - checkpoint metadata
    - checkpoint loading

- `train_conditional_family_severity.sh`
  - Example shell launcher.

Most important change in your trainer:

Old:

```python
edited, delta_obj, delta_bg, attention, effective_delta = model(images)
```

New:

```python
severity_norm, codec_family_id = get_conditions_for_batch(
    args=args,
    codec_family_to_id=codec_family_to_id,
    images=images,
    batch=batch,
)

edited, delta_obj, delta_bg, attention, effective_delta = model(
    images,
    severity=severity_norm,
    codec_family=codec_family_id,
)
```

Severity normalization:

- If severity is already normalized:
  - `--severity-min 0 --severity-max 1`

- If raw severity is QP-like and larger means worse:
  - `--severity-min <min_qp> --severity-max <max_qp>`

- If raw severity is JPEG-quality-like and larger means better:
  - add `--invert-severity`
