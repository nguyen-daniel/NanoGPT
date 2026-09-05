# DirectML val loss is not ground truth

DirectML can train on this Windows + AMD box. The reported validation loss on DirectML cannot be trusted as a measure of the saved weights.

## The 1.47 vs 5.86 episode

A 5000-iter **6×6×384** run executed on the RX 7800 XT (`privateuseone:1`). Numbers are from merged PR #6 / commit `5d3d217` (`results/train_run.json` on that commit), and the local CPU checkpoint still on this machine matches the CPU side of that writeup.

| Number | Value | Trusted? |
|--------|-------|----------|
| DirectML-reported best val | **1.4659** | No. Device-side eval of a long DML run. |
| CPU re-eval of those same DML weights | **5.8566** | Yes. Host copy of the `state_dict`. |
| CPU-gated DML save (best host val) | **3.17** at iter 250 | Yes, as a gate. Training on DML then stopped improving on CPU around unigram/bigram. |
| CPU-trained 6×6×192 (the sample) | **1.709** at iter 4750 | Yes. `out_cpu/ckpt.pt` on this machine: `iter_num=4750`, `best_val_loss=1.7092256546020508`, vocab 65, ~47 min (`elapsed_sec=2810.2`). |

The Shakespeare-like `ROMEO:` sample in `results/sample_shakespeare.txt` is from the **CPU** 6×6×192 run, not from the DirectML 6×6×384 weights.

## What to do

- Train on DirectML if you want the 7800 XT. That path is real (`results/gpu_proof.json`).
- **Always re-eval on CPU** before you believe a val number or keep a checkpoint:

```bash
python train.py --eval_only --eval_device cpu --checkpoint out/ckpt.pt
# or, after a train:
python train.py --device directml --cpu_eval --no_amp --no_compile
```

- Do not treat a falling DirectML val (1.47) as proof the weights are a 1.47-loss language model. On this stack they were a 5.86-loss model when scored on CPU.
- New checkpoints store the char `vocab` list so a 65-char pre-UNK run is not decoded with today's 66+UNK `data/vocab.pt`.
