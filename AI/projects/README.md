# Projects

Source: https://www.deep-ml.com/projects

---

## 1. Attention Is All You Need: Build the Transformer From Scratch
**Difficulty**: Hard | **Framework**: PyTorch | **Steps**: 79

Reimplement the original encoder-decoder Transformer end to end in PyTorch, from token vocabularies and sinusoidal positional encodings through multi-head attention, label smoothing, Noam scheduling, and beam search.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. Tokenization and Batching | 6 | Vocabulary, encode/decode ids, pad and batch sequences |
| 2. Embeddings and Positional Encoding | 7 | Scale embeddings, build sinusoidal positional encoding matrix |
| 3. Masks and Scaled Dot-Product Attention | 9 | Padding mask, causal mask, raw scores → softmax → weighted values |
| 4. Multi-Head Attention | 9 | Split/merge heads, project Q/K/V, assemble full MHA module |
| 5. Feed-Forward, LayerNorm, and Dropout | 7 | Position-wise FFN, layer norm, residual add-and-norm, dropout |
| 6. Encoder, Decoder, and Full Model | 13 | Stack encoder/decoder layers, tie embeddings, run full forward pass |
| 7. Parameter Initialization | 4 | Allocate weight tensors for encoder/decoder layers and embeddings |
| 8. Training Objective and Schedule | 8 | Teacher forcing, Noam LR, label-smoothed KL loss, token accuracy |
| 9. Adam Optimizer From Scratch | 6 | Moment buffers, EMA updates, bias correction, parameter step |
| 10. Training Step and Loop | 3 | Forward pass, backprop, full training loop |
| 11. Decoding and Beam Search | 7 | Greedy argmax, length penalty, beam expansion, best beam selection |

---

## 2. Build a Mini LLM Inference Server
**Difficulty**: Hard | **Stack**: ML Systems | **Steps**: 51

Construct a complete LLM inference stack from scratch: sampling, tokenization, a tiny transformer with KV cache, paged memory, and a continuous-batching scheduler.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. Sampling Primitives | 6 | Stable softmax, temperature scaling, top-k/top-p filters, stochastic sampling |
| 2. Tokenization | 3 | Tiny vocabulary, encode/decode between strings and token id sequences |
| 3. Tiny Transformer with KV Cache | 7 | Embeddings, linear projections, KV cache, causal attention, prefill/decode |
| 4. Paged KV Cache | 9 | Block-based KV allocator, allocation/free, paged appends, paged attention |
| 5. Sequences and Static Batching | 8 | Per-request sequence state, single-sequence generation, batched sequences |
| 6. Continuous Batching and Scheduling | 8 | Capacity checks, priority queues, admission, preemption, mixed prefill/decode |
| 7. Serving API | 5 | Request/response interface, streaming chunks, submission, driver loop |
| 8. Benchmarking | 5 | TTFT, inter-token latency, throughput, latency percentiles |

---

## 3. Build a Trainable CNN from Scratch in NumPy
**Difficulty**: Hard | **Steps**: 59

Assemble a LeNet-style convolutional network entirely in NumPy, from numerically stable softmax through im2col convolutions, backprop, and a full training loop.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. Softmax, Loss, and Metrics Primitives | 9 | Numerically stable softmax, cross-entropy loss, accuracy helpers |
| 2. Initialization and Convolution Plumbing | 7 | He init, zero biases, padding, output-shape math, im2col/col2im |
| 3. Layer Forward and Backward Passes | 17 | Conv, max pooling, ReLU, flatten, and linear — forward and backward |
| 4. Fused Loss and Optimizers | 8 | Softmax-CE fusion, SGD and Adam parameter updates |
| 5. Assembling LeNet | 10 | Compose conv/classifier blocks, full LeNet forward and backward pass |
| 6. Synthetic Data Pipeline | 4 | Generate dataset, shuffle, train/test split, minibatch iterator |
| 7. Training Loop and Evaluation | 4 | Training step, epoch loop, training driver, held-out evaluation |

---

## 4. DiLoCo: Distributed Low-Communication Training of Language Models
**Difficulty**: Hard | **Steps**: 30

Build DiLoCo from scratch: workers train locally with AdamW for many inner steps, then a Nesterov-momentum outer optimizer synchronizes the models with a single all-reduce.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. Model, Forward & Loss | 6 | Parameter init, forward pass, activations, softmax, cross-entropy |
| 2. Inner Optimizer: AdamW | 5 | Per-worker AdamW: state init, moment updates, bias correction, step |
| 3. Parameter Arithmetic Utilities | 4 | Clone, scale, subtract, average — pytree-style parameter operations |
| 4. Data Sharding & Local Worker Training | 5 | IID/non-IID partitioning, per-worker batches, local training loop |
| 5. Outer Optimizer with Nesterov Momentum | 4 | Server-side optimizer: momentum buffers, Nesterov parameter update |
| 6. DiLoCo Training Loop & Baseline | 3 | Full communication round, outer loop across rounds, baseline comparison |
| 7. Evaluation & Communication Analysis | 3 | Held-out loss, accuracy, communication savings vs all-reduce baseline |

---

## 5. Flash Attention in CUDA from Scratch
**Difficulty**: Hard | **Stack**: CUDA | **Steps**: 26 | Runs on NVIDIA T4 GPU

Build a tiled, IO-aware Flash Attention implementation in CUDA, starting from elementary GPU primitives up to a causal fused kernel.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. CUDA Primitives Warm-up | 5 | Elementwise and reduction kernels for matrix and attention ops |
| 2. Matrix Operations | 3 | Matmul, transpose, dot-product utilities for attention scoring |
| 3. Naive Attention Baseline | 4 | QKᵀ scoring, row-wise softmax, PV multiplication |
| 4. Online Softmax Math | 4 | Running-max and running-sum updates for incremental softmax across tiles |
| 5. Tiled Attention Building Blocks | 6 | Per-tile shared-memory routines: load, score, reduce, exponentiate, accumulate |
| 6. Fused Flash Attention Kernel | 2 | Assemble tiled blocks into the full Flash Attention kernel + host launcher |
| 7. Causal Flash Attention | 2 | Extend kernel with a causal mask for autoregressive attention |

---

## 6. LoRA Fine-Tune a Tiny Chat Model with Unsloth
**Difficulty**: Easy | **Stack**: Unsloth | **Steps**: 20

Build an end-to-end LoRA fine-tuning pipeline for a 4-bit Qwen2.5-0.5B chat model using Unsloth.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. Load the Quantized Base Model | 4 | Load 4-bit Qwen2.5, inspect parameter count, verify quantization, fix padding token |
| 2. Attach LoRA Adapters | 4 | Select target modules, wrap with LoRA, measure trainable parameter count |
| 3. Build the Instruction Dataset | 6 | Create instruction/response pairs, format into training strings, build Dataset, split |
| 4. Run the SFT Training Loop | 3 | Configure TrainingArguments, build SFTTrainer, run optimization steps |
| 5. Inference with the Tuned Model | 3 | Switch to inference mode, build chat-template prompt, generate and decode reply |

---

## 7. Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy
**Difficulty**: Hard | **Steps**: 40

Build a complete training stack in pure NumPy that mirrors how modern frameworks scale models: gradient accumulation, activation checkpointing, mixed precision, data parallelism, and ZeRO sharding.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. MLP Forward and Backward Core | 10 | Data setup, parameters, forward pass, loss, manual backprop for two-layer MLP |
| 2. Gradient Accumulation | 4 | Split batches into micro-batches, accumulate gradients under memory constraint |
| 3. Activation Checkpointing | 4 | Recompute activations during backward to trade compute for memory |
| 4. Mixed Precision Training | 6 | Forward/backward in fp16, full-precision master weights, loss scaling |
| 5. Data Parallel Training | 6 | Shard data across workers, compute local gradients, synchronize with all-reduce |
| 6. ZeRO-Style Optimizer Sharding | 5 | Partition Adam optimizer state and params across workers, all-gather to reconstruct |
| 7. Memory Accounting and Full Training Loop | 5 | Quantify model/optimizer/activation memory, run the full combined training loop |

---

## 8. RAG Pipeline
**Difficulty**: Hard | **Framework**: PyTorch | **Steps**: 51

Construct a complete RAG system step by step, from raw document ingestion and chunking through embeddings, dense retrieval, prompting, advanced retrieval, and evaluation.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. Document Ingestion & Preprocessing | 5 | Load text from files/HTML/directories, normalize, wrap into document objects |
| 2. Chunking Strategies | 5 | Fixed-size, token-based, sentence-aware, and overlapping chunkers |
| 3. Embeddings & Corpus Storage | 5 | Load sentence-transformer, embed and normalize chunks, persist to disk |
| 4. Dense Retrieval with NumPy and FAISS | 8 | Cosine similarity from scratch, FAISS index, correctness verification |
| 5. Prompting and Answer Generation | 9 | Grounded prompts, local instruct model, generate answers, source citations |
| 6. Advanced Retrieval Techniques | 8 | Query rewriting, HyDE, BM25, hybrid search, cross-encoder reranking |
| 7. Evaluation | 6 | Eval set, hit rate, recall@k, MRR, faithfulness metrics |
| 8. Robustness, Caching, and Chat Memory | 5 | Abstention, deduplication, embedding caching, conversational memory |

---

## 9. Tiny GPT From Scratch
**Difficulty**: Hard | **Framework**: NumPy (PyTorch version also available) | **Steps**: 166

Build a small character-level GPT end-to-end in pure NumPy, from tokenization and array basics all the way to a trained, text-generating model.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. Tokenizer | 7 | Character-level tokenizer with vocab, stoi/itos, encode/decode |
| 2. NumPy and Softmax Foundations | 26 | Arrays, indexing, broadcasting, reductions, numerically stable softmax |
| 3. Data Pipeline and Bigram Baseline | 23 | Corpus loading, batched (X,Y) sequences, counting-based bigram model |
| 4. Single-Layer Neural Bigram | 17 | Learned weight matrix, cross-entropy derivation, gradients, SGD update |
| 5. Layer Primitives and Backprop | 18 | Forward/backward for linear, bias, ReLU, softmax+CE, LayerNorm |
| 6. Embeddings and Self-Attention | 39 | Token/positional embeddings, masked single-head and multi-head attention |
| 7. FFN, Blocks, and Full Model | 16 | Feed-forward networks, residual connections, pre-LN Transformer blocks |
| 8. Adam, Training Loop, and Generation | 20 | Adam optimizer, training/validation loop, temperature sampling, top-k generation |

---

## 10. Trainable Mixture of Experts in CUDA
**Difficulty**: Hard | **Stack**: CUDA | **Steps**: 52 | Runs on NVIDIA T4 GPU

Build a trainable Mixture-of-Experts layer from scratch in CUDA, starting from low-level matmul and activation kernels up to a full MoE forward/backward with load balancing.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. CUDA Building Blocks | 13 | Core CUDA primitives: matmul variants, bias add, reductions, activations |
| 2. Top-K Gating Utilities | 3 | Select top-k gate values per token, normalize gate weights |
| 3. Router | 4 | Gate matmul → probabilities → top-k expert selection per token |
| 4. Token Dispatch and Combine | 8 | Token counts per expert, slot offsets, scatter/gather for sparse routing |
| 5. Expert MLP Forward and Backward | 12 | Two-layer expert MLP with up/down projections, forward and backward |
| 6. Load Balancing Auxiliary Loss | 4 | Dispatch fractions, mean router probabilities, auxiliary loss |
| 7. Loss, Optimizer, and Training Loop | 8 | MSE task loss, gradient zeroing, SGD kernel, full training orchestration |

---

## 11. Vision-Language Model from Scratch in PyTorch
**Difficulty**: Hard | **Framework**: PyTorch | **Steps**: 62

Build an end-to-end multimodal vision-language model that ingests an image plus a text prompt and autoregressively generates a caption.

| Part | Steps | What you build |
|------|-------|----------------|
| 1. Patch Embedding and ViT Input | 6 | Raw image → patch embeddings, class token, learnable positional embeddings |
| 2. Multi-Head Self-Attention | 13 | Scaled dot-product attention from primitives, full MHA module |
| 3. MLP, Normalization, and Encoder Blocks | 10 | Position-wise MLP, layer norm, pre-norm residual sublayers, ViT encoder stack |
| 4. Vision-to-Language Projector | 4 | Extract patch features, project into language model's embedding dimension |
| 5. Text Embedding and Multimodal Fusion | 8 | Tokenizer, text embeddings, splice projected image tokens into sequence |
| 6. Causal Decoder and Forward Pass | 7 | Masked transformer decoder, language modeling head, full VLM forward pass |
| 7. Loss and Caption Generation | 8 | Next-token cross-entropy with masking, temperature/top-k sampling |
| 8. Training Loop | 6 | Parameter init, autograd training steps, gradient descent, full training run |
