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

