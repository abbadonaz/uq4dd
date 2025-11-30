1. Add:
- a new dataset config for your own data(with censor info)
- a new encoder config if you use a different molecular representation

2. Extend the model side:
- uses the Student-t/NIG PDF for uncensored points 
- uses CDF - based terms for left censored points 
- keeps the evidential regularizer 

Re-use the existing model=evidential flag with a config switch use_censoring=true, or we can introduce model = evidential_censored 


Overall:
- use current implementation CensoredEvidentialLoss for comparison
- implement a combination of TobitLoss pattern 




INFO they alredy implemented a class: **CensoredEvidentialLoss**

- in evidential regression (Amini et al, 2020) the network outputs four parameters:
- gamma - the predicted mean, what model believes 
- v - the evidence or "precision scaling" parameter 
- alpha - the shape parameter 
- beta - the scale parameter 

Together they define a Normal Inverse Gamma distribution 

In this repository - **CensoredEvidentialLoss** : 

- they have a masking convention:
    mask == 0 uncensored
    mask == 1 right censored (true value ≥ label)
    mask == -1 left-censored (true value ≤ label)

- they compute  vlilation indication: 

"""
diff = target - gamma
# right-censored: clamp positive errors (gamma > target) to 0
diff[mask == 1] = clamp(min=0)
# left-censored: clamp negative errors (gamma < target) to 0
diff[mask == -1] = clamp(max=0)
diff = diff ** 2
correct = (mask != 0) & (diff == 0)
""" 

- if censored sample is consistent with censor sign - no loss
- for censored "but incorrect" points - punish with usual evidential loss as if labels were exact 

My change: ** ExtendedCensoredEvidentialLoss** 

