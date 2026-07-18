# StandardScaler Fit and Transform - Q&A

---

## Core Idea

**Q: What does StandardScaler compute?**
A: Training mean and standard deviation per feature.

**Q: What is the formula?**
A: `(x - mean) / std`.

**Q: Where should scaler fit happen?**
A: Training data only.

## Usage

**Q: Which models often need scaling?**
A: Distance-based, gradient-based, SVM, PCA, and regularized linear models.
