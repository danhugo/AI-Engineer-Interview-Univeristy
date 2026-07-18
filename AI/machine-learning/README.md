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
- [ ] [Gradient Descent Variants (Batch / SGD / Mini-batch)](optimization-training/gradient-descent-variants/note.md)
- [ ] [Adam / AdamW Optimizer Step](optimization-training/adam-adamw-optimizer-step/note.md)
- [ ] [Gradient Clipping](optimization-training/gradient-clipping/note.md)
- [ ] [Weight Decay vs L2 Regularization](optimization-training/weight-decay-vs-l2-regularization/note.md)
- [ ] [Dropout (forward + inference mode)](optimization-training/dropout/note.md)
- [ ] [Batch Normalization Forward Pass](optimization-training/batch-normalization-forward-pass/note.md)
- [ ] [Feature Scaling](optimization-training/feature-scaling/note.md)
- [ ] [Early Stopping Based on Validation Loss](optimization-training/early-stopping-validation-loss/note.md)
- [ ] [Mixed Precision Training](optimization-training/mixed-precision-training/note.md)
- [ ] [Gradient Checkpointing](optimization-training/gradient-checkpointing/note.md)

## 4. Learning Rate Schedulers
- [ ] [StepLR](learning-rate-schedulers/step-lr/note.md)
- [ ] [ExponentialLR](learning-rate-schedulers/exponential-lr/note.md)
- [ ] [CosineAnnealingLR](learning-rate-schedulers/cosine-annealing-lr/note.md)
- [ ] [Cosine Annealing with Warm Restarts](learning-rate-schedulers/cosine-annealing-warm-restarts/note.md)
- [ ] [Linear Warmup Schedule](learning-rate-schedulers/linear-warmup-schedule/note.md)
- [ ] [Warmup + Cosine Decay](learning-rate-schedulers/warmup-cosine-decay/note.md)

## 5. Model Validation & Selection
- [ ] [K-Fold Cross-Validation](model-validation-selection/k-fold-cross-validation/note.md)
- [ ] [Stratified Train-Test Split](model-validation-selection/stratified-train-test-split/note.md)
- [ ] [Bias-Variance Decomposition from Bootstrap](model-validation-selection/bias-variance-decomposition-bootstrap/note.md)
- [ ] [Grid Search](model-validation-selection/grid-search/note.md)
- [ ] [Random Search vs Grid Search](model-validation-selection/random-search-vs-grid-search/note.md)
- [ ] [Bayesian Hyperparameter Optimization](model-validation-selection/bayesian-hyperparameter-optimization/note.md)
- [ ] [Learning Curve Generator for Bias-Variance Diagnosis](model-validation-selection/learning-curve-bias-variance-diagnosis/note.md)

## 6. Preprocessing
- [ ] [One-Hot Encoding](preprocessing/one-hot-encoding/note.md)
- [ ] [Label Encoding](preprocessing/label-encoding/note.md)
- [ ] [StandardScaler Fit and Transform](preprocessing/standardscaler-fit-transform/note.md)
- [ ] [MinMaxScaler](preprocessing/minmaxscaler/note.md)
- [ ] [Missing Value Imputation](preprocessing/missing-value-imputation/note.md)
- [ ] [Train/Val/Test Split with No Leakage](preprocessing/train-val-test-split-no-leakage/note.md)

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
- [ ] [Gini Impurity](decision-trees-ensembles/gini-impurity/note.md)
- [ ] [Entropy-based Split Selection](decision-trees-ensembles/entropy-based-split-selection/note.md)
- [ ] [Decision Tree Classification](decision-trees-ensembles/decision-tree-classification/note.md)
- [ ] [Decision Tree Regression](decision-trees-ensembles/decision-tree-regression/note.md)
- [ ] [Random Forest Fit from Scratch](decision-trees-ensembles/random-forest-fit-from-scratch/note.md)
- [ ] [Random Forest Feature Importance](decision-trees-ensembles/random-forest-feature-importance/note.md)
- [ ] [Out-of-Bag Score](decision-trees-ensembles/out-of-bag-score/note.md)
- [ ] [Bagging Classifier from Scratch](decision-trees-ensembles/bagging-classifier-from-scratch/note.md)
- [ ] [AdaBoost Fit Method](decision-trees-ensembles/adaboost-fit-method/note.md)
- [ ] [Gradient Boosting Regressor Step](decision-trees-ensembles/gradient-boosting-regressor-step/note.md)
- [ ] [XGBoost Objective Function](decision-trees-ensembles/xgboost-objective-function/note.md)
- [ ] [LightGBM Objective Function](decision-trees-ensembles/lightgbm-objective-function/note.md)

## 10. Support Vector Machines
- [ ] [Linear Kernel](support-vector-machines/linear-kernel/note.md)
- [ ] [RBF Gaussian Kernel](support-vector-machines/rbf-gaussian-kernel/note.md)
- [ ] [Polynomial Kernel](support-vector-machines/polynomial-kernel/note.md)
- [ ] [Sigmoid Kernel](support-vector-machines/sigmoid-kernel/note.md)
- [ ] [SVM Margin Width](support-vector-machines/svm-margin-width/note.md)
- [ ] [Hinge Loss for SVM](support-vector-machines/hinge-loss-for-svm/note.md)
- [ ] [Pegasos Kernel SVM](support-vector-machines/pegasos-kernel-svm/note.md)

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
- [ ] [Layer Normalization for Sequence Data](nlp-llm/layer-normalization-sequence-data/note.md)
- [ ] [Sparse Window Attention](nlp-llm/sparse-window-attention/note.md)
- [ ] [GPT FeedForward Block (Linear-GELU-Linear)](nlp-llm/gpt-feedforward-block/note.md)
- [ ] [Temperature Sampling](nlp-llm/temperature-sampling/note.md)
- [ ] [Beam Search Decoding](nlp-llm/beam-search-decoding/note.md)
- [ ] [ROUGE Score](nlp-llm/rouge-score/note.md)
- [ ] [Extend BPE Tokenizer](nlp-llm/extend-bpe-tokenizer/note.md)
- [ ] [Chat Template Encoding](nlp-llm/chat-template-encoding/note.md)
- [ ] [Count Trainable Parameters (Weight Tying)](nlp-llm/count-trainable-parameters-weight-tying/note.md)
- [ ] [Model Memory Footprint](nlp-llm/model-memory-footprint/note.md)
- [ ] [Shortcut Connection Gradient Effect](nlp-llm/shortcut-connection-gradient-effect/note.md)
- [ ] [Multi-Head Attention from Scratch](nlp-llm/multi-head-attention-from-scratch/note.md)
- [ ] [Rotary Positional Encoding (RoPE)](nlp-llm/rotary-positional-encoding-rope/note.md)
- [ ] [Grouped Query Attention (GQA)](nlp-llm/grouped-query-attention-gqa/note.md)
- [ ] [RMSNorm](nlp-llm/rmsnorm/note.md)
- [ ] [LoRA / QLoRA Parameter Count](nlp-llm/lora-qlora-parameter-count/note.md)
- [ ] [Flash Attention (concept + memory complexity)](nlp-llm/flash-attention-concept-memory-complexity/note.md)
- [ ] [BLEU Score](nlp-llm/bleu-score/note.md)
- [ ] [Perplexity Calculation](nlp-llm/perplexity-calculation/note.md)

## 16. LLM Serving & Inference
- [ ] [Tokens-per-Second Throughput](llm-serving-inference/tokens-per-second-throughput/note.md)
- [ ] [Compute TTFT, ITL, TPS from Token Timestamp Stream](llm-serving-inference/compute-ttft-itl-tps-token-timestamps/note.md)
- [ ] [End-to-End Latency Decomposition](llm-serving-inference/end-to-end-latency-decomposition/note.md)
- [ ] [Prefix Cache Hit Rate](llm-serving-inference/prefix-cache-hit-rate/note.md)
- [ ] [KV Cache Tiered Offloading](llm-serving-inference/kv-cache-tiered-offloading/note.md)
- [ ] [N-gram Speculation Dictionary Construction](llm-serving-inference/ngram-speculation-dictionary-construction/note.md)
- [ ] [Speculative Decoding Acceptance Rate vs Temperature](llm-serving-inference/speculative-decoding-acceptance-rate-vs-temperature/note.md)
- [ ] [Expert Parallelism Token Routing and Communication Cost](llm-serving-inference/expert-parallelism-token-routing-communication-cost/note.md)
- [ ] [Disaggregated Prefill-Decode Serving](llm-serving-inference/disaggregated-prefill-decode-serving/note.md)
- [ ] [Cold Start Latency Budget Breakdown](llm-serving-inference/cold-start-latency-budget-breakdown/note.md)
- [ ] [Break-Even Pay-Per-Token API vs Dedicated GPU](llm-serving-inference/break-even-pay-per-token-api-vs-dedicated-gpu/note.md)
- [ ] [Continuous Batching Throughput](llm-serving-inference/continuous-batching-throughput/note.md)
- [ ] [PagedAttention KV Cache Utilization](llm-serving-inference/pagedattention-kv-cache-utilization/note.md)
- [ ] [LoRA Adapter Hot-Swap Overhead](llm-serving-inference/lora-adapter-hot-swap-overhead/note.md)

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
