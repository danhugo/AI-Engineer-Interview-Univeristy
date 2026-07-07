# Deep Learning Problems

Problems from [deep-ml.com](https://www.deep-ml.com/problems?category=Deep+Learning), grouped by topic.

---

## Activation Functions

- [ ] Sigmoid Activation Function Understanding (easy)
- [ ] Softmax Activation Function Implementation (easy)
- [ ] Implementation of Log Softmax Function (easy)
- [ ] Implement ReLU Activation Function (easy)
- [ ] Leaky ReLU Activation Function (easy)
- [ ] Implement the Hard Sigmoid Activation Function (easy)
- [ ] Implement the ELU Activation Function (easy)
- [ ] Implement PReLU Forward and Backward Pass (medium)
- [ ] Implement the Softplus Activation Function (easy)
- [ ] Implement the Softsign Activation Function (easy)
- [ ] Implement the Swish Activation Function (easy)
- [ ] Implement the SELU Activation Function (easy)
- [ ] GeLU Activation Function (easy)
- [ ] Implement the Mish Activation Function (easy)
- [ ] Implement the Tanh Activation Function (easy)
- [ ] Implement the Hardtanh Activation Function (easy)
- [ ] Implement the Square ReLU Activation Function (easy)
- [ ] Dynamic Tanh: Normalization-Free Transformer Activation (easy)
- [ ] SwiGLU activation function (easy)
- [ ] Derivatives of Activation Functions (easy)

---

## Core Network Layers & Architecture

- [ ] Single Neuron (easy)
- [ ] Single Neuron with Backpropagation (medium)
- [ ] Implementing a Custom Dense Layer in Python (hard)
- [ ] Simple Convolutional 2D Layer (medium)
- [ ] Implement a Dense Block with 2D Convolutions (hard)
- [ ] Overlapping Max Pooling (medium)
- [ ] Implement 2D Average Pooling (easy)
- [ ] Implement Global Average Pooling (easy)
- [ ] Local Response Normalization (LRN) (medium)
- [ ] Implement a Simple Residual Block with Shortcut Connection (easy)
- [ ] Implement a Sequential Container for Neural Network Layers (easy)
- [ ] Implement FlattenConsecutive Layer for Hierarchical Fusion (medium)
- [ ] Auto-Sized MLP from Gym Spaces (easy)
- [ ] Implement Module with Register Buffer for Causal Mask (easy)
- [ ] 3D CNN Forward Pass Implementation (hard)
- [ ] Build Scaled Dot-Product Attention (medium)
- [ ] Build a Transformer Encoder Layer (hard)
- [ ] TDNN for Variable-Length Sequences (hard)
- [ ] Implement Graph Convolution Network (GCN) Layer (medium)
- [ ] Implement a Spatiotemporal Transformer Block (hard)

---

## Normalization

- [ ] Implement Batch Normalization for BCHW Input (medium)
- [ ] Implement Group Normalization (medium)
- [ ] Instance Normalization (IN) Implementation (medium)
- [ ] Implement RMSNorm (Root Mean Square Layer Normalization) (easy)
- [ ] BatchNorm1d Forward with Bessel's Correction (easy)
- [ ] BatchNorm1d Supporting 2D and 3D Inputs (medium)
- [ ] Fused Backward Pass of BatchNorm1d (hard)
- [ ] Implement Bias-less Cohere LayerNorm (easy)
- [ ] Bias-less LayerNorm with Float32 Compute (easy)
- [ ] Gemma-Style RMSNorm with Zero-Centered Scale (easy)
- [ ] Gemma-Style RMSNorm with Optional Scale (easy)
- [ ] Zero-Initialized RMSNorm with (1+w) Scaling (easy)

---

## Optimizers

- [ ] Implement Adam Optimization Algorithm (medium)
- [ ] Adam Optimizer (medium)
- [ ] Adagrad Optimizer (easy)
- [ ] Momentum Optimizer (easy)
- [ ] Adamax Optimizer (easy)
- [ ] Adadelta Optimizer (medium)
- [ ] Nesterov Accelerated Gradient Optimizer (easy)
- [ ] Muon Optimizer Update with Newton-Schulz Iteration (medium)
- [ ] Implement Gradient Clipping by Value (easy)
- [ ] Checkpoint Averaging (Polyak Averaging) (easy)

---

## Loss Functions

- [ ] KL Divergence Between Two Normal Distributions (easy)
- [ ] Compute Multi-class Cross-Entropy Loss (easy)
- [ ] Knowledge Distillation Loss (medium)
- [ ] Implement Binary Cross-Entropy Loss (easy)
- [ ] Contrastive Loss (InfoNCE / SimCLR-style) (medium)
- [ ] Triplet Margin Loss (medium)
- [ ] Implement Variational Autoencoder (VAE) Loss (ELBO) (medium)
- [ ] Weighted Multi-Task Loss for Joint Diffusion and Trajectory Learning (medium)
- [ ] AlphaZero Policy and Value Loss (easy)
- [ ] PTX Loss for Catastrophic Forgetting Prevention (RLHF) (medium)
- [ ] Masked Cross-Entropy Loss for Supervised Finetuning (medium)
- [ ] DPO Loss with NLL Regularization on Chosen Sequences (medium)
- [ ] DPO Implicit Reward Computation (easy)
- [ ] Noise Prediction Loss for Diffusion Training (medium)

---

## Backpropagation & Autograd

- [ ] Implementing Basic Autograd Operations (medium)
- [ ] Implement a Simple CNN Training Function with Backpropagation (hard)
- [ ] Implement a Simple RNN with Backpropagation Through Time (BPTT) (hard)
- [ ] TD Error Backpropagation in Networks (medium)
- [ ] Backpropagation Through a Scalar Chain Network (medium)
- [ ] Backpropagation Gradients for a Dense Layer (medium)
- [ ] Manual Backprop Through Cross-Entropy Intermediates (hard)
- [ ] Reduce Gradient Across Broadcasted Dimensions (easy)
- [ ] Backward Pass of Tanh Using Cached Output (easy)
- [ ] Backward Pass of Tensor Reshape (easy)
- [ ] Backward Pass of Row-wise Max Operation (medium)
- [ ] Compute the Hessian Matrix (medium)
- [ ] Stop-Gradient Operator in Jointly Trained Models (easy)

---

## Recurrent Networks (RNN / LSTM / GRU)

- [ ] Implementing a Simple RNN (medium)
- [ ] Implement Long Short-Term Memory (LSTM) Network (medium)
- [ ] Implement GRU Cell (medium)
- [ ] Implement Neural Memory Update with Surprise and Momentum (medium)

---

## Attention & Transformers

- [ ] Implement Self-Attention Mechanism (medium)
- [ ] Implement Masked Self-Attention (medium)
- [ ] Implement Multi-Head Attention (hard)
- [ ] Positional Encoding Calculator (hard)
- [ ] Implement Position-wise Feed-Forward Block with Residual and Dropout (medium)
- [ ] Flash Attention v1 - Forward Pass (hard)
- [ ] Implement Multi-Head Attention Using Einsum (hard)
- [ ] Multi-Head Attention via Head Stacking (medium)
- [ ] Multi-Head Attention with Combined QKV Weight Matrix (medium)
- [ ] Implement Batched Causal Self-Attention (medium)
- [ ] Apply Dropout to Attention Weights (easy)
- [ ] Compare Naive vs Stable Softmax for Attention Scores (easy)
- [ ] Scale Attention Scores by sqrt(d_k) (easy)
- [ ] Construct Causal Attention Mask via tril and triu Methods (easy)
- [ ] Batched Attention Score Computation for Multiple Heads (medium)
- [ ] Simple Self-Attention Without Trainable Weights (easy)
- [ ] Transfer Weights from Linear-Style to Parameter-Style Self-Attention (medium)
- [ ] Sliding Window Attention (medium)
- [ ] Build Causal Mask with Position Offsets for Cached Attention (medium)
- [ ] Implement Gated Attention (medium)
- [ ] Implement Gated DeltaNet Linear Attention (hard)
- [ ] Delta Rule Update for Associative Memory (medium)
- [ ] Hybrid Linear/Full Attention Layer Routing (medium)
- [ ] Parallel Transformer Block Forward Pass (medium)
- [ ] Pre-Norm GPT Transformer Block Forward Pass (hard)
- [ ] Pre-Norm vs Post-Norm Transformer Block (medium)
- [ ] Parallel Attention and FFN Transformer Block (medium)
- [ ] Implement Cross-Layer KV Sharing in Transformer (hard)

---

## Positional Embeddings

- [ ] Character-Level Tokenizer (stoi/itos/BOS) (easy)
- [ ] Learned Positional Embeddings (easy)
- [ ] Rotary Positional Embeddings (RoPE) (medium)
- [ ] Implement Llama 3.1 RoPE Frequency Rescaling (hard)
- [ ] Llama 3 RoPE Frequency Scaling (hard)
- [ ] NoPE (No Positional Embedding) with iRoPE Attention (hard)
- [ ] Partial Rotary Position Embedding (Partial RoPE) (medium)
- [ ] Proportional RoPE Inverse Frequencies with NoPE Tail (medium)
- [ ] Token Embedding Lookup Table (easy)
- [ ] Add Positional Embeddings to Token Embeddings (easy)
- [ ] Embedding Layer as One-Hot Matrix Multiplication (easy)
- [ ] Scaled Token Embedding Lookup (easy)
- [ ] Implement Weight Tying Between Embedding and LM Head (easy)

---

## KV Cache & Efficient Inference

- [ ] KV Cache for Efficient Autoregressive Attention (medium)
- [ ] Pre-allocated Sliding KV Cache Update (hard)
- [ ] KV Cache Size Estimator for MLA vs MHA vs GQA (easy)
- [ ] KV Cache Estimator with Sliding Window Attention (medium)
- [ ] KV Cache Memory Budget and Eviction Policy (medium)
- [ ] Truncate KV Cache for Sliding Window Attention (medium)
- [ ] Sliding Window KV Cache Truncation (medium)
- [ ] Implement Speculative Decoding Verification (hard)
- [ ] Classifier-Free Guidance Skip Speedup Calculator (medium)
- [ ] Estimate KV-Cache Memory for MHA vs Linear Attention (easy)
- [ ] Estimate KV-Cache Memory with GQA and Cross-Layer Sharing (medium)

---

## Quantization & Precision

- [ ] Implement INT8 Quantization (medium)
- [ ] Block-wise FP8 Quantization (medium)
- [ ] Post-Training Quantization with Per-Channel Scale Factors (medium)
- [ ] FP4 Quantization with Microscaling (MXFP4) (hard)
- [ ] Number Format Precision Comparison (FP16 vs BF16 vs FP8 vs FP4) (hard)
- [ ] Row-wise FP8 Quantization with Clamped Scale Factors (medium)
- [ ] Implement Weight Tying Between Embedding and LM Head (easy)

---

## Mixture of Experts (MoE)

- [ ] Calculate Computational Efficiency of MoE (easy)
- [ ] Implement the Noisy Top-K Gating Function (medium)
- [ ] Implement a Sparse Mixture of Experts Layer (hard)
- [ ] Sparse MoE Top-K Routing (medium)
- [ ] Mixture of Experts Load Balancing Loss (medium)
- [ ] MoE with Shared Expert Forward Pass (medium)
- [ ] Hash-Based Expert Routing for MoE Layers (easy)
- [ ] Anticipatory Routing for MoE Training Stability (hard)
- [ ] Implement Sigmoid MoE Router with Bias Correction (hard)
- [ ] MoE Parameter and Active-Token Memory Estimator (medium)
- [ ] Estimate MoE vs Dense FFN Parameters with Match-Dense Sizing (medium)
- [ ] Train a Paris-Style Decentralized Expert Model (medium)

---

## Multi-Query / Grouped-Query Attention

- [ ] Implement Multiquery Attention (MQA) (medium)
- [ ] Implement Grouped Query Attention (GQA) (medium)
- [ ] Multi-Head Latent Attention (MLA) (hard)
- [ ] QK-Norm (Query-Key Normalization) (medium)

---

## Weight Initialization & Regularization

- [ ] Implement Xavier/Glorot Weight Initialization (medium)
- [ ] Implement He Weight Initialization (easy)
- [ ] Implement He Weight Initialization for Neural Networks (medium)
- [ ] Xavier/Glorot Weight Initialization (easy)
- [ ] Spectral Normalization (medium)
- [ ] Dropout Layer (medium)
- [ ] Regularization via Information Bottleneck (easy)

---

## Fine-Tuning & Adaptation (LoRA / Adapters)

- [ ] LoRA: Low-Rank Adaptation Forward Pass (medium)
- [ ] QLoRA: Quantized Low-Rank Adaptation Forward Pass (medium)
- [ ] Merge LoRA Adapter Weights for Zero-Latency Inference (easy)
- [ ] Calculate LoRA Trainable Parameter Count vs Full Fine-Tuning (easy)
- [ ] LoRA-Adapted Linear Projection with Task Switching (medium)
- [ ] Efficient Task Switching with LoRA Weight Swapping (medium)
- [ ] Implement Adapter Bottleneck Layer for Transformer Fine-Tuning (medium)
- [ ] Calculate Memory Savings of LoRA vs Full Fine-Tuning with Adam (medium)
- [ ] BitFit: Bias-Only Parameter Selection for Fine-Tuning (easy)
- [ ] Simulate LoRA Rank Selection with Synthetic Low-Rank Updates (medium)
- [ ] Recursively Replace Linear Layers with LoRA (medium)

---

## Diffusion Models

- [ ] KL Divergence Between Two Normal Distributions (easy)
- [ ] Diffusion Reconstruction Loss (medium)
- [ ] Forward Diffusion Process (medium)
- [ ] Forward & Backward Diffusion Process (medium)
- [ ] DDPM Noise Schedule (Linear Beta Schedule) (medium)
- [ ] Implement DDPM Reverse Sampling Step (medium)
- [ ] Classifier-Free Guidance for Conditional Diffusion (medium)
- [ ] DDIM Deterministic Sampling Step (medium)
- [ ] Diffusion Model U-Net Time Embedding (medium)
- [ ] Exponential Moving Average (EMA) for Diffusion Model Weights (easy)
- [ ] Latent Diffusion Encoding and Decoding (medium)
- [ ] Diffusion Cosine Noise Schedule (medium)
- [ ] Implement Score Matching for Score-Based Diffusion (medium)
- [ ] Implement Relativistic Critic Rewards for Adversarial Reasoning (medium)
- [ ] Training a Rectified Flow Diffusion Model (hard)
- [ ] Euler Sampling for Rectified Flow Models (medium)
- [ ] Logit-Normal Sampling for Diffusion Timesteps (easy)
- [ ] UniPC Predictor-Corrector Step (medium)

---

## Transformers for Vision & Video

- [ ] Frame-wise Causal Attention Masking for Video Transformers (medium)
- [ ] Guidance Attention Mask for Chunked Video (medium)
- [ ] Sequential Video Generation with Diffusion Models (hard)
- [ ] Multi-Hypothesis Trajectory Prediction (medium)
- [ ] Multi-term Memory Patchification for Video (medium)
- [ ] First Frame Anchor Noise Injection (easy)
- [ ] Unified History Injection for Autoregressive Video Diffusion (medium)
- [ ] Context Parallelism with Ring Attention for Video Models (hard)
- [ ] VLM Visual Token Count from Image Resolution and Patch Size (easy)
- [ ] Video Generation Latent Space Memory Estimation (medium)
- [ ] Action-Conditioned Autoregressive Video Dynamics Model (hard)

---

## Generative Models (VAE / VQ-VAE / GAN / World Models)

- [ ] Train a Simple GAN on 1D Gaussian Data (hard)
- [ ] Build a VQ-VAE from Scratch (hard)
- [ ] Vector Quantization with Straight-Through Gradients (medium)
- [ ] EMA Codebook Updates in VQ-VAE (medium)
- [ ] Masked Generative Token Prediction Step (hard)
- [ ] Iterative Token Unmasking with MaskGIT Decoding (medium)
- [ ] Unsupervised Latent Action Discovery (hard)
- [ ] Goal-Conditioned World Model with Future Anchoring (hard)
- [ ] Adaptive Layer Normalization for Conditional Generation (medium)
- [ ] Multi-Signal Conditioning for Diffusion Transformers (easy)
- [ ] Additive vs. Concatenated Action Conditioning in Transformers (medium)

---

## Reinforcement Learning

- [ ] Deep Q-Network Implementation (hard)
- [ ] Target Network Update for Stable Learning (easy)
- [ ] Prioritized Experience Replay (medium)
- [ ] Dueling Network Architecture (medium)
- [ ] DQN NatureCNN Architecture Analysis (easy)
- [ ] Distributed On-Policy Reinforcement Learning (hard)
- [ ] Implement RAFT for RLVR with Binary Rewards (medium)
- [ ] Implement RAFT++ with Importance Sampling and Clipping (medium)
- [ ] Implement Reinforce-Rej: Filtering Trivial Prompt Groups (medium)

---

## Systems: Parallelism & Distributed Training

- [ ] Tensor Parallelism All-Reduce Communication Cost (medium)
- [ ] Designing 4D Parallelism for Large Model Training (hard)
- [ ] Simulate DDP All-Reduce Gradient Averaging (medium)
- [ ] Pipeline Parallelism Bubble Ratio Calculator (easy)
- [ ] Computing Optimal Model Size with Scaling Laws (medium)
- [ ] Distribute SWA and Full Attention Layers by Ratio (easy)
- [ ] Deterministic Sparse Attention Backward KV Accumulation (hard)
- [ ] Deterministic MoE Backward with Buffer Isolation (hard)
- [ ] Storage-Pointer-Based Recomputation Deduplication (medium)
- [ ] Interleaved CSA-HCA Layer Assignment (easy)

---

## Systems: Memory & Serving (vLLM / PagedAttention)

- [ ] Implementing PagedAttention: Block-wise Attention Computation (hard)
- [ ] Building a Virtual Memory System for KV Cache (medium)
- [ ] Copy-on-Write Memory Sharing for LLM Sampling (medium)
- [ ] Analyzing Memory Fragmentation in LLM Serving (easy)
- [ ] Preemption Strategies: Swapping vs Recomputation (medium)
- [ ] Micro-batch Inference Throughput Estimation (easy)
- [ ] Model FLOPs Utilization (MFU) Calculator (easy)

---

## Language Model Internals

- [ ] Train Bigram Language Model as a Neural Network (medium)
- [ ] MLP Character-Level Language Model Forward Pass (medium)
- [ ] Generate Input-Target Batches for Language Model Training (easy)
- [ ] Causal Cumulative Mean via Triangular Matrix Multiply (medium)
- [ ] Swap LM Head with Classification Head (easy)
- [ ] Last-Token Classification from Sequence Logits (easy)

---

## Miscellaneous / Advanced Topics

- [ ] The Pattern Weaver's Code (medium)
- [ ] Implement the SGTM Parameter Update Step (medium)
- [ ] Mean Ablation for Circuit Discovery (medium)
- [ ] Inference Head Pruning for Transformers (medium)
- [ ] Temperature Decay Scheduler (medium)
- [ ] Calculate Number of Parameters in Neural Network (medium)
- [ ] Distance Correlation for Measuring Metadata Dependence (medium)
- [ ] MDN with Label Collinearity Control (hard)
- [ ] Implement mHC Forward Pass (medium)
- [ ] Implement Core MDN Residualization (hard)
- [ ] Engram Context-Aware Gating (medium)
- [ ] Dynamic Parameter Generation for Hyper-Connections (medium)
- [ ] EAGLE-Style Draft Model from Hidden States (medium)
- [ ] Thanksgiving Feast Predictor: Softmax for Dish Selection (easy)
