# Standalone conditional pre-editor

This package contains a completely standalone trainer.

It does not call:
- `run_attention_experiment.py`
- `run_attention_experiment_rateconditioned.py`
- `run_attention_experiment_realcodecs_family_ablation.py`
- any previous file

Files:
- `train_preedit_attention_conditional.py`
- `train_conditional_family_severity.sh`

Run:

```bash
chmod +x train_conditional_family_severity.sh
./train_conditional_family_severity.sh
```

Conditioning:
- codec family is embedded using `nn.Embedding`
- severity is normalized to `[0, 1]`
- both are injected with FiLM modulation inside the U-Net

Default codec families:
- `diff_jpeg`
- `jp2_like`
- `bmshj`
- `bpg_like`

Default severity:
- already normalized
- levels: `0.0,0.25,0.5,0.75,1.0`

Checkpoints:
- `runs/preedit_conditional_family_severity_coco2017/last.pt`
- `runs/preedit_conditional_family_severity_coco2017/best_loss.pt`
