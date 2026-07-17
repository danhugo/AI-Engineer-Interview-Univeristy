# Machine Learning — Essential Problems

Trimmed to the core set that keeps showing up in ML interviews.
Each one teaches a concept you must know.

---

## 1. Linear & Logistic Regression
- [x] Linear & Logistic Regression — Study Sheet
- [ ] [Linear Regression — Normal Equation](linear-logistic-regression/linear-regression-normal-equation/note.md)
- [ ] [Linear Regression — Gradient Descent](linear-logistic-regression/linear-regression-gradient-descent/note.md)
- [ ] [Ridge Regression Loss](linear-logistic-regression/ridge-regression-loss/note.md)
- [ ] [Lasso Regression via ISTA](linear-logistic-regression/lasso-regression-ista/note.md)
- [ ] [Elastic Net Regression via Gradient Descent](linear-logistic-regression/elastic-net-gradient-descent/note.md)
- [ ] [Polynomial Regression Fit](linear-logistic-regression/polynomial-regression-fit/note.md)
- [ ] [Binary Classification with Logistic Regression](linear-logistic-regression/binary-logistic-regression/note.md)
- [ ] [Train Logistic Regression with Gradient Descent](linear-logistic-regression/logistic-regression-gradient-descent/note.md)
- [ ] [Train Softmax Regression with Gradient Descent](linear-logistic-regression/softmax-regression-gradient-descent/note.md)

## 2. Loss Functions
- [ ] [MSE Loss](loss-function/mse-loss/note.md)
- [ ] [Cross-Entropy Loss](loss-function/cross-entropy-loss/note.md)
- [ ] [Hinge Loss (SVM)](loss-function/hinge-loss/note.md)
- [ ] [Huber Loss](loss-function/huber-loss/note.md)
- [ ] [Focal Loss (imbalanced classification)](loss-function/focal-loss/note.md)
- [ ] [Label Smoothing for Multi-Class Cross-Entropy](loss-function/label-smoothing-cross-entropy/note.md)
- [ ] [KL Divergence](loss-function/kl-divergence/note.md)
- [ ] [Contrastive Loss (SimCLR / metric learning)](loss-function/contrastive-loss/note.md)
- [ ] [Triplet Loss](loss-function/triplet-loss/note.md)

## 3. Optimization & Training
- [ ] Gradient Descent Variants (Batch / SGD / Mini-batch)
- [ ] Adam / AdamW Optimizer Step
- [ ] Gradient Clipping
- [ ] Weight Decay vs L2 Regularization
- [ ] Dropout (forward + inference mode)
- [ ] Batch Normalization Forward Pass
- [ ] Feature Scaling
- [ ] Early Stopping Based on Validation Loss
- [ ] Mixed Precision Training
- [ ] Gradient Checkpointing

## 4. Learning Rate Schedulers
- [ ] StepLR
- [ ] ExponentialLR
- [ ] CosineAnnealingLR
- [ ] Cosine Annealing with Warm Restarts
- [ ] Linear Warmup Schedule
- [ ] Warmup + Cosine Decay

## 5. Model Validation & Selection
- [ ] K-Fold Cross-Validation
- [ ] Stratified Train-Test Split
- [ ] Bias-Variance Decomposition from Bootstrap
- [ ] Grid Search
- [ ] Random Search vs Grid Search
- [ ] Bayesian Hyperparameter Optimization
- [ ] Learning Curve Generator for Bias-Variance Diagnosis

## 6. Preprocessing
- [ ] One-Hot Encoding
- [ ] Label Encoding
- [ ] StandardScaler Fit and Transform
- [ ] MinMaxScaler
- [ ] Missing Value Imputation (mean / median / mode)
- [ ] Train/Val/Test Split with No Leakage

## 7. Classification Metrics
- [ ] Accuracy Score
- [ ] Precision
- [ ] Recall
- [ ] F1 Score (Binary)
- [ ] Macro / Micro / Weighted F1 Score
- [ ] Top-K Accuracy for Multi-Class
- [ ] Confusion Matrix
- [ ] ROC Curve & AUC
- [ ] Precision-Recall Curve
- [ ] Jaccard Index
- [ ] Dice Score

## 8. Regression Metrics
- [ ] RMSE
- [ ] MAE
- [ ] R-squared
- [ ] MAPE (Mean Absolute Percentage Error)

## 9. Decision Trees & Ensembles
- [ ] Gini Impurity
- [ ] Entropy-based Split Selection
- [ ] Decision Tree (Classification)
- [ ] Decision Tree (Regression)
- [ ] Random Forest Fit from Scratch
- [ ] Random Forest Feature Importance
- [ ] Out-of-Bag Score
- [ ] Bagging Classifier from Scratch
- [ ] AdaBoost Fit Method
- [ ] Gradient Boosting Regressor Step
- [ ] XGBoost Objective Function
- [ ] LightGBM Objective Function

## 10. Support Vector Machines
- [ ] Linear Kernel
- [ ] RBF (Gaussian) Kernel
- [ ] Polynomial Kernel
- [ ] Sigmoid Kernel
- [ ] SVM Margin Width
- [ ] Hinge Loss for SVM
- [ ] Pegasos Kernel SVM

## 11. Naive Bayes
- [ ] Gaussian Naive Bayes
- [ ] Multinomial Naive Bayes
- [ ] Bernoulli Naive Bayes

## 12. K-Nearest Neighbors
- [ ] K-Nearest Neighbors

## 13. Clustering
- [ ] K-Means
- [ ] K-Means++ Initialization
- [ ] DBSCAN
- [ ] BIRCH Clustering
- [ ] Hierarchical (Agglomerative) Clustering
- [ ] Gaussian Mixture Model with EM
- [ ] Silhouette Score
- [ ] Elbow Method for K Selection

## 14. Dimensionality Reduction
- [ ] Principal Component Analysis (PCA)
- [ ] Explained Variance Ratio for PCA
- [ ] Linear Discriminant Analysis (LDA)
- [ ] t-SNE Gradient
- [ ] UMAP
- [ ] Kernel PCA

## 15. NLP / LLM
- [ ] Layer Normalization for Sequence Data
- [ ] Sparse Window Attention
- [ ] GPT FeedForward Block (Linear-GELU-Linear)
- [ ] Temperature Sampling
- [ ] Beam Search Decoding
- [ ] ROUGE Score
- [ ] Extend BPE Tokenizer
- [ ] Chat Template Encoding
- [ ] Count Trainable Parameters (Weight Tying)
- [ ] Model Memory Footprint
- [ ] Shortcut Connection Gradient Effect
- [ ] Multi-Head Attention from Scratch
- [ ] Rotary Positional Encoding (RoPE)
- [ ] Grouped Query Attention (GQA)
- [ ] RMSNorm
- [ ] LoRA / QLoRA Parameter Count
- [ ] Flash Attention (concept + memory complexity)
- [ ] BLEU Score
- [ ] Perplexity Calculation

## 16. LLM Serving & Inference
- [ ] Tokens-per-Second Throughput
- [ ] Compute TTFT, ITL, TPS from Token Timestamp Stream
- [ ] End-to-End Latency Decomposition
- [ ] Prefix Cache Hit Rate
- [ ] KV Cache Tiered Offloading
- [ ] N-gram Speculation Dictionary Construction
- [ ] Speculative Decoding Acceptance Rate vs Temperature
- [ ] Expert Parallelism Token Routing and Communication Cost
- [ ] Disaggregated Prefill-Decode Serving
- [ ] Cold Start Latency Budget Breakdown
- [ ] Break-Even Pay-Per-Token API vs Dedicated GPU
- [ ] Continuous Batching Throughput
- [ ] PagedAttention KV Cache Utilization
- [ ] LoRA Adapter Hot-Swap Overhead

## 17. Scaling Laws & RLHF
- [ ] Power-Law Scaling Law
- [ ] Compute-Optimal Model Size (IsoFLOP)
- [ ] Sigmoidal Accuracy-to-Log-Likelihood Scaling Law Fit
- [ ] KL-Penalized Reward Shaping (RLHF)
- [ ] RLAIF Reward Model
- [ ] Combined RM and LLM Quality Filter
- [ ] DPO (Direct Preference Optimization) Loss
- [ ] PPO Clipped Objective
- [ ] Reward Model Calibration
