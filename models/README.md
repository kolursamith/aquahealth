# models/

Trained checkpoints (`final_model.pth`) are produced by `src/train.py` and
are **not** committed initially.

`.gitignore` excludes `models/**/*.pth` — once a final checkpoint exists,
decide whether it belongs in Git (via Git LFS, given GitHub's file size
limits) or should be distributed separately (release asset, shared drive,
cloud storage).

Expected layout:

```
models/
└── final_model.pth
```
