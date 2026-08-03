# Building a GPT from Scratch

**Aaryan Patil** · Started July 24, 2026

Building a GPT implemented from first principles. PyTorch provides tensors and autograd; everything above that line is hand-written.

---

## The Idea

I'm building a transformer by building each component by hand, verifying it works, then use
it as a primitive for the next one. Nothing gets stacked on top of a black box.

That constraint sets the build order — you can't hand-write attention on top of an
MLP you didn't hand-write.


## Quick Start

```bash
pip install -r requirements.txt
python -m foundations.training_loop
```

---

## Roadmap

- [x] Math foundations — gradient descent, activations, loss functions
- [x] Neural networks from scratch — neuron, backprop, MLP
- [ ] NLP pipeline — BPE tokenizer, embeddings, batched loader
- [ ] Attention — single head, causal masking, multi-head
- [ ] Transformer block — residuals, feed-forward, normalization variants
- [ ] GPT model + autoregressive text generation
- [ ] KV-cache and grouped-query attention
- [ ] Custom CUDA kernels for the attention hot path
- [ ] Quantization (INT8 / INT4 weight-only)
- [ ] Throughput benchmarking vs. HuggingFace baseline

**External dependencies:** PyTorch for tensor ops and autograd. That's the line.
