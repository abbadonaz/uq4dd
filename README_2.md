📘 Censored Evidential Regression in This Repository

This repository extends Deep Evidential Regression (Amini et al., 2020) to handle left- and right-censored regression labels common in drug discovery (e.g., “<1 µM”, “>10 µM”).

The base uq4dd repository already includes an implementation called CensoredEvidentialLoss. This project introduces an improved, likelihood-correct version called ExtendedCensoredEvidentialLoss.

This section documents:

How evidential regression works

How uq4dd originally handled censored labels

What this repository adds

How to use the improved censored evidential loss

1. 🔍 Background: Evidential Regression (Amini et al., 2020)

Deep Evidential Regression predicts the parameters of a Normal–Inverse-Gamma (NIG) distribution:

γ (gamma) – predicted mean

ν (v) – evidence / precision scaling

 # Censored Evidential Regression

This repository extends Deep Evidential Regression (Amini et al., 2020) to handle left- and right-censored regression labels commonly found in molecular property prediction and drug discovery (for example, “< 1 µM” and “> 10 µM”).

The original `uq4dd` project included a class named `CensoredEvidentialLoss`. This repo adds an improved, likelihood-correct variant: `ExtendedCensoredEvidentialLoss` (aka `ProperCensoredEvidentialLoss`).

The sections below cover:

- Background on evidential regression
- The original `CensoredEvidentialLoss` and its limitations
- The `ExtendedCensoredEvidentialLoss` approach and why it is preferred
- How to enable/use it in `uq4dd`

---

Background: Evidential Regression

Evidential regression predicts the parameters of a Normal–Inverse–Gamma (NIG) distribution. The network head emits four outputs per sample:

- $\gamma$ — predicted mean
- $\nu$ — evidence / precision scaling
- $\alpha$ — shape
- $\beta$ — scale

These parameters imply a Student-t predictive distribution. They also permit decomposition of predictive uncertainty into aleatoric and epistemic components:

Aleatoric variance:
$$
\sigma_{\text{aleatoric}}^2 = \frac{\beta}{\alpha - 1}
$$

Epistemic variance:
$$
\sigma_{\text{epistemic}}^2 = \frac{\beta}{\alpha - 1}\cdot\frac{1}{\nu}
$$

Thus the model head must output $(\gamma,\nu,\alpha,\beta)$ per sample.

---

Original `CensoredEvidentialLoss` (uq4dd)

`uq4dd` represents censoring with an `operator` mask:

- `0`: uncensored (exact target)
- `+1`: right-censored (true value $\ge$ label)
- `-1`: left-censored (true value $\le$ label)

The original loss implements a clamp/hinge on the prediction error. In effect:

- If a censored sample's prediction lies on the allowed side of the bound, it contributes zero loss.
- If the prediction violates the bound, that sample is treated as if it were uncensored and receives the full evidential loss.

This heuristic encourages predictions to satisfy censor directions but does not use censored samples to shape uncertainty.

### Key limitations

- Does not use the Student-t PDF for uncensored samples.
- Does not use Student-t CDF tail probabilities to score censored samples.
- Zero loss for any censored point that satisfies its bound → no gradient on uncertainty parameters for those samples.
- Not statistically consistent with censored-likelihood (Tobit-style) approaches.

As a consequence, censored data contribute little to learning aleatoric/epistemic uncertainty.

---

Extended / Proper Censored Evidential Loss

`ExtendedCensoredEvidentialLoss` implements a different censored likelihood using the Student-t predictive PDF and CDF derived from $(\gamma,\nu,\alpha,\beta)$.

For a sample with observation $y$, Student-t PDF $p_t(\cdot)$ and CDF $F(\cdot)$:

- Uncensored (operator = 0):
	$$\mathcal{L}_{\text{data}} = -\log p_t(y)$$

- Left-censored (operator = -1, threshold $L$):
	$$\mathcal{L}_{\text{data}} = -\log \mathbb{P}(Y \le L) = -\log F(L)$$

- Right-censored (operator = +1, threshold $U$):
	$$\mathcal{L}_{\text{data}} = -\log \mathbb{P}(Y \ge U) = -\log (1 - F(U))$$

Additionally the loss includes the evidential regularizer (for example):
$$
\mathcal{L}_{\text{reg}} = |y - \gamma| \cdot (2\nu + \alpha)
$$

and the final sample loss is the sum (or weighted sum) of the data likelihood term and the regularizer.

### Benefits

- Fully probabilistic and statistically consistent with Tobit-style likelihoods.
- Smooth gradients for censored samples, so uncertainty parameters are learned from censoring information.
- Censored samples influence both mean and uncertainty estimates.

---


---

Integration & Usage

No architecture changes are required — the evidential head stays the same. Example wiring (pseudocode in `DeepDTI.__init__`):

```python
if predictor.name == "Evidential":
		if censored:
				self.criterion = ExtendedCensoredEvidentialLoss()
```

Hydra config snippet:

tbd

---

## 6. Testing & Validation

Included are synthetic tests and real-data examples for:

- Left-censored datasets
- Right-censored datasets
- Mixed censored + uncensored datasets

Comparisons are made versus:
- Original `CensoredEvidentialLoss`
- `TobitLoss` (Gaussian censored likelihood)
- Uncensored `EvidentialLoss`

Validation checks include predictive mean behavior, aleatoric/epistemic decomposition, and correct CDF-tail behaviour on censored samples.

---

---

Author / contact: see repository metadata

This repository replaces the  censored evidential loss from uq4dd with a proper, likelihood-based censored evidential loss built on the Student-t predictive distribution implied by NIG parameters.

The result is a statistically sound and fully differentiable approach for modeling censored regression data with evidential uncertainty — particularly important in molecular property prediction where censoring is common.