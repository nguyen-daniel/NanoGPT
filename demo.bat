@echo off
REM Windows one-command demo (Make is optional). Same path as `make demo`.
python scripts/download_demo_ckpt.py --out out_demo/ckpt.pt
if errorlevel 1 exit /b 1
python sample.py --checkpoint out_demo/ckpt.pt --prompt "ROMEO:" --num_tokens 200 --temperature 0.8 --top_k 40 --device cpu
