# Polynomial Regression Fit — Q&A

---

## Intuition

**Q: What is polynomial regression?**
A: Polynomial regression is linear regression after feature expansion. It turns `x` into features like `[1, x, x^2, x^3]`.

**Q: What is the main trick in polynomial regression?**
A: The model can draw a curved line by adding powers of `x`. But it still fits ordinary linear weights.

**Q: Is polynomial regression nonlinear or linear?**
A: It is nonlinear in `x`, but linear in the learned weights.

---

## 1. Why It Is Still Linear Regression

**Q: When is a model considered linear?**
A: A model is linear when it is linear in the learned weights.

**Q: Why is `theta0 + theta1*x + theta2*x^2` linear in the weights?**
A: The learned weights are not squared or multiplied together. They only multiply fixed input features.

**Q: Does polynomial regression need a new optimizer?**
A: No. It uses ordinary least squares on the expanded feature matrix.

---

## 2. Polynomial Features

**Q: What features do we create for one input and degree 3?**
A: `[1, x, x^2, x^3]`.

**Q: Why include a column of ones?**
A: It lets the model learn an intercept or bias term.

**Q: What is an interaction term?**
A: An interaction term combines features, like `ab` for inputs `a` and `b`. It lets the model learn that two features matter together.

---

## 3. Fitting With Least Squares

**Q: What objective does polynomial regression minimize?**
A: It minimizes the squared error between `Phi @ theta` and `y`.

**Q: How do you fit polynomial regression in NumPy?**
A: Build the polynomial feature matrix `Phi`, then call `np.linalg.lstsq(Phi, y, rcond=None)[0]`.

**Q: Why not compute the inverse directly?**
A: Least-squares solvers are more numerically stable than manually computing an inverse.

---

## 4. Prediction

**Q: How do you predict on new values?**
A: Build the same polynomial features for the new values, then compute `Phi_new @ theta`.

**Q: What is the most common prediction bug?**
A: Using a different feature order during prediction than during training.

**Q: If training uses `[1, x, x^2]`, what must prediction use?**
A: The exact same feature order: `[1, x, x^2]`.

---

## 5. Degree Controls Flexibility

**Q: What does degree mean?**
A: Degree is the highest power included in the feature matrix.

**Q: What happens when the degree is high?**
A: The curve can bend more. It may fit training data well but overfit.

**Q: What are signs of overfitting?**
A: Very low training error, high validation or test error, and a curve that wiggles between points.

---

## 6. Scaling

**Q: Why does scaling matter for polynomial regression?**
A: Powers of large inputs can become huge. That can make fitting unstable.

**Q: What should you do before creating high-degree polynomial features?**
A: Scale the input features.

---

## 7. Complexity

**Q: What is the feature creation cost for one input feature?**
A: For `n` samples and degree `d`, it costs `O(nd)`.

**Q: What happens with many input features?**
A: The number of polynomial and interaction features grows quickly.

**Q: What is usually the expensive part?**
A: Least-squares fitting after the feature matrix is built.

---

## 8. Interview Gotchas

**Q: What is the shortest interview explanation of polynomial regression?**
A: It is linear regression on polynomial features.

**Q: How should you handle the bias term?**
A: Include it consistently as a column of ones or let the library handle it.

**Q: What solver should you use in interview code?**
A: Use `np.linalg.lstsq`, not a manual matrix inverse.
