conda activate lerobot-dynamicVLA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless

hf download lerobot/metaworld_mt50 \
  --repo-type dataset \
  --local-dir datasets/metaworld_mt50

hf download tiantianx/pi05_metaworld \
  --local-dir pretrained_weights/pi05_metaworld

python scripts/make_pi05_compat.py \
  --source pretrained_weights/pi05_metaworld \
  --output pretrained_weights/pi05_metaworld_compat

scripts/extract_laco_metaworld.sh
scripts/train_laco_metaworld.sh
scripts/calibrate_laco_metaworld.sh
scripts/eval_laco_metaworld.sh

scripts/eval_laco_metaworld.sh configs/laco/metaworld_pi05.yaml \
  --set evaluation.mode=baseline \
  --set evaluation.output=outputs/laco/eval_baseline

scripts/eval_laco_metaworld.sh configs/laco/metaworld_pi05.yaml \
  --set evaluation.mode=replan \
  --set evaluation.output=outputs/laco/eval_replan

scripts/eval_laco_metaworld.sh configs/laco/metaworld_pi05.yaml \
  --set evaluation.mode=laco \
  --set evaluation.output=outputs/laco/eval_laco