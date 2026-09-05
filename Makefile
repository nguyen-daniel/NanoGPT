.PHONY: data train sample demo bench bench-cpu smoke lint test

PYTHON ?= python
DEMO_CKPT ?= out_demo/ckpt.pt

data:
	$(PYTHON) data.py

train:
	$(PYTHON) train.py --max_iters 50 --eval_interval 50 --eval_iters 2 --no_compile --no_amp

sample:
	$(PYTHON) sample.py --num_tokens 50

# Recruiter path: download the CPU 6x6x192 ckpt if missing, print ROMEO: text.
# Does not train. Weights live on the demo-ckpt-cpu-6x6x192 GitHub Release.
demo:
	$(PYTHON) scripts/download_demo_ckpt.py --out $(DEMO_CKPT)
	$(PYTHON) sample.py --checkpoint $(DEMO_CKPT) --prompt "ROMEO:" --num_tokens 200 --temperature 0.8 --top_k 40 --device cpu

bench:
	$(PYTHON) benches/bench_attention.py
	$(PYTHON) benches/bench_dataloader.py
	$(PYTHON) benches/bench_bpe.py

bench-cpu:
	$(PYTHON) benches/bench_attention.py --device cpu
	$(PYTHON) benches/bench_dataloader.py --device cpu
	$(PYTHON) benches/bench_bpe.py

smoke:
	$(PYTHON) -c "from train import train; train(max_iters=5, eval_interval=5, eval_iters=1, use_compile=False, use_amp=False, device='cpu')"

lint:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m unittest discover -s tests -v
