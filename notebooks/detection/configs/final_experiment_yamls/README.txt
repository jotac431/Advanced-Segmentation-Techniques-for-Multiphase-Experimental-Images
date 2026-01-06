YAMLs generated to match your plan (augmentation only analyzed after transfer).

Phases / files:
1) Synthetic sweep (already done):
   - maskrcnn_resnet50_fpn_synth.yaml
   - maskrcnn_resnet50_fpn_v2_synth.yaml
   - maskrcnn_resnet101_fpn_synth.yaml

2) Real-only baseline (NO AUG):
   - maskrcnn_resnet50_fpn_real.yaml
   - maskrcnn_resnet50_fpn_v2_real.yaml
   - maskrcnn_resnet101_fpn_real.yaml

3) Transfer baseline (NO AUG):
   - maskrcnn_resnet50_fpn_transfer_real.yaml
   - maskrcnn_resnet50_fpn_v2_transfer_real.yaml
   - maskrcnn_resnet101_fpn_transfer_real.yaml

4) Augmentation phase (files include '_aug' only here):
   - maskrcnn_resnet50_fpn_real_aug.yaml
   - maskrcnn_resnet50_fpn_v2_real_aug.yaml
   - maskrcnn_resnet101_fpn_real_aug.yaml
   - maskrcnn_resnet50_fpn_transfer_real_aug.yaml
   - maskrcnn_resnet50_fpn_v2_transfer_real_aug.yaml
   - maskrcnn_resnet101_fpn_transfer_real_aug.yaml

Required edits:
- For transfer v2 and r101: update transfer.init_checkpoint placeholders.
- Everything else keeps your original dataset paths and training hyperparameters.

Augmentation control:
- augmentation.enabled: false => true NoAug (only resize + dtype + tensor)
- augmentation.enabled: true  => augmentation policy active
- Only augmentation-phase YAMLs enable it.
