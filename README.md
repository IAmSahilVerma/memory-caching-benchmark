# Memory Caching vs Transformer: A Sequence Recall Benchmark

A small-scale empirical replication of the core claim behind Google Research / Cornell's **Memory Caching** paper (ICLR 2026 submission): that RNNs augmented with cached hidden states can close the recall-performance gap with Transformers, at a fraction of the compute cost.

This project trains a Transformer and three GRU variants (baseline, Residual Memory Caching, Gated Residual Memory Caching) from scratch on three tasks designed to stress different kinds of sequence recall, then compares them under matched parameter counts and matched training hyperparameters.

**[View the full interactive report](report/index.html)** (open directly in a browser, no server required)

---

## Background

Transformers dominate sequence modelling because attention gives every token full visibility into the entire sequence, at the cost of quadratic compute. RNNs are far cheaper, but compress everything into a fixed-size hidden state, so they tend to forget information from earlier in the sequence.

Memory Caching proposes a middle ground: periodically save snapshots of an RNN's hidden state as it processes a sequence, then let later timesteps look back at those cached snapshots instead of relying solely on the current (lossy) hidden state. The paper proposes several variants of this idea, the two relevant to this project being:

- **Residual Memory Caching (RMC)** — sum all cached states and add them to the current hidden state. No learned parameters.
- **Gated Residual Memory Caching (GRMC)** — learn an input-dependent gate per cached segment before summing, so the model can weight relevant past segments more heavily than irrelevant ones.

This project asks: at small, reproducible scale, does this actually work?

---

## What's tested

| Model | Description |
|---|---|
| `transformer` | Small GPT-style causal Transformer (4 layers, 4 heads) — upper-bound reference |
| `gru` | Plain 2-layer GRU — lower-bound reference, no caching |
| `gru_rmc` | GRU + Residual Memory Caching |
| `gru_grmc` | GRU + Gated Residual Memory Caching |

| Task | What it tests |
|---|---|
| **Selective Copy** | Recall specific marked values scattered throughout a long sequence, reproduced autoregressively after a copy marker (formulation follows Gu & Dao, *Mamba*, 2023) |
| **Associative Recall** | Recall a value from a key-value pair seen earlier, given only the key — true content-based lookup, not just positional memory |
| **Language Modelling** | Standard next-token prediction on WikiText-2 (GPT-2 BPE tokenizer), grounding the synthetic results in a real-world signal |

---

## Results summary

| Model | Selective Copy (acc) | Associative Recall (acc) | Language Modelling (perplexity) |
|---|---|---|---|
| Transformer | **99.2%** | **11.3%** | 500.2 |
| GRU (baseline) | 24.1% | 3.75% | **294.4** |
| GRU + RMC | 18.3% | **8.75%** | 311.8 |
| GRU + GRMC | **25.3%** | 3.5% | 300.7 |

**Key findings:**

- On **Selective Copy**, GRMC clearly outperforms both the plain GRU and RMC, supporting the paper's central claim that gating helps. RMC's validation loss was still rising at convergence even as its accuracy plateaued — a sign of training instability that the gated variant didn't show.
- On **Associative Recall**, the ordering flips: RMC outperforms GRMC and the baseline. This is the hardest task for every recurrent model, and the result suggests the value of gating vs. plain caching may depend on the type of recall required, not just model scale.
- On **Language Modelling**, the plain GRU baseline achieves the lowest perplexity of the four. Natural language rewards general statistical fluency more than long-range recall at this scale, so the extra caching machinery shows no advantage here.

Full per-epoch training curves and a detailed discussion are in the [interactive report](report/index.html).

---

## Methodology notes (read this before trusting the numbers)

This project went through two rounds of fixing real fairness issues, both documented here rather than quietly patched, because they materially changed the results:

1. **Parameter count.** An early version gave the Transformer ~3.8x more parameters than the GRU variants on the synthetic tasks. All results above use parameter-matched models (within 0.3% of each other per task).
2. **Learning rate.** An early version used a higher learning rate for GRU variants than the Transformer, to work around a slow training start. All results above use one learning rate (3e-4) and one optimizer configuration for every model.
3. **Training budget.** Once learning rate was unified, the GRU variants on Selective Copy needed 80 epochs (not 40) to actually converge — confirmed by checking that their loss/accuracy curves had flattened, not just stopped improving within an arbitrary epoch budget. The Transformer converges within ~20 epochs and is shown at 40.

A few additional things worth knowing if you read the code:

- An earlier version of `GRUBaseline`/`GRUWithMemoryCache` used orthogonal initialization on GRU input weights, which suppressed gradient signal almost entirely and caused training to flatline. Switched to PyTorch's default GRU init.
- The original Selective Copy task design asked the model to fill 8 identical, unordered query slots in parallel — which is information-theoretically unsolvable, since nothing distinguishes one query slot from another. Redesigned as an autoregressive recall task (matching the original Mamba paper's formulation), where each recalled value is fed back in to predict the next.

---

## Repository structure

```
memory-caching-benchmark/
├── models/
│   ├── transformer.py      # SmallTransformer
│   ├── gru.py               # GRUBaseline
│   └── gru_mc.py             # GRUWithMemoryCache (RMC + GRMC variants)
├── tasks/
│   ├── selective_copy.py
│   ├── associative_recall.py
│   └── language_model.py    # WikiText-2 via GPT-2 BPE tokenizer
├── train.py                  # Model-agnostic training loop
├── evaluate.py                # Experiment runner across all model x task combinations
├── results/
│   └── all_results.json       # Per-epoch metrics for every model x task run
├── report/
│   └── index.html              # Standalone interactive report (Chart.js, no server needed)
└── requirements.txt
```

---

## Running it yourself

```bash
pip install -r requirements.txt
python -m evaluate
```

This runs the full sweep (4 models x 3 tasks) and writes results to `results/all_results.json`. Training is small enough to run on a single consumer GPU (developed and tested on an RTX 4050, ~6GB VRAM); the full sweep takes roughly 1-2 hours depending on epoch budget.

To regenerate the report after a new run, the contents of `results/all_results.json` need to be inlined into `report/index.html` (see the `const RESULTS = ...` assignment near the bottom of the file).

---

## Relationship to the original paper

This is a small-scale, independent replication, not a reproduction. The original paper evaluates Titans-style backbone architectures up to 1.3B parameters on Multi-Query Associative Recall (MQAR) and standard language modelling benchmarks. This project uses a plain GRU (not Titans) at roughly 800K-1.5M parameters, and a custom single-query Associative Recall task rather than MQAR. The qualitative direction of the headline result (gating helps on the task it's most suited to) is consistent with the paper's own ablations; the magnitude and consistency across tasks should not be taken as a direct comparison to their reported numbers.

---

## Citation

Behrouz et al., *Memory Caching: RNNs with Growing Memory*, submitted to ICLR 2026. Google Research, Cornell University, USC.
