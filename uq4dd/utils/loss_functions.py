
import math
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn import GaussianNLLLoss
from torch.distributions import StudentT
class CensoredMSELoss(_Loss):
    '''
    Re-implementation of the CensoredMSE loss proposed by Arany et al., (2022).
    '''
    def __init__(self, absolute=False, reduction='mean') -> None:
        super().__init__()
        assert reduction in ['none', 'mean', 'root'], 'Reduce in censored loss should be none, mean or root'
        self.absolute = absolute
        self.reduction = reduction
        if self.absolute: 
            print('CensoredMAE have not been proparly tested yet, make sure implementation is correct before use.')

    def forward(self, input, target):
        mask = target['Operator']
        diff = target['Label'] - input
        diff[mask == 1] = torch.clamp(diff[mask == 1], min=0.0) # Clamp to min 0, same as max(target-input, 0)
        diff[mask == -1] = torch.clamp(diff[mask == -1], max=0.0) # Clamp to max 0, same as min(target-input, 0)
        if not self.absolute: 
            diff = diff ** 2
        else: 
            diff = torch.abs(diff)
        if self.reduction == 'mean':
            return torch.mean(diff)
        elif self.reduction == 'root': 
            return torch.sqrt(torch.mean(diff))
        else: 
            return diff


class TobitLoss(_Loss): 
    '''
    Our proposed extension of the Gaussian NLL loss to handle censored data.
    '''
    def __init__(self, reduction='mean') -> None:
        super().__init__()
        assert reduction in ['none', 'mean'], 'Reduce in censored loss should be none, mean or root'
        self.reduction = reduction
        self.nll_loss = GaussianNLLLoss()
        self.eps = 1e-6

    def forward(self, input, target, var):
        mask = target['Operator']
        label = target['Label']
        
        # Loss for observed points, i.e. -log(phi(y)) if mask == 0
        loss = (1 - torch.abs(mask)) * self.nll_loss(input, label, var)
        
        # Loss for censored points, i.e. -log(1 - Phi(y)) if mask == 1, - log(Phi(y)) if mask == -1
        if len(input[mask == 1]) > 0: 
            normal_right = torch.distributions.normal.Normal(input[mask == 1], torch.sqrt(var[mask == 1]))
            loss[mask == 1] = - torch.log(1 - normal_right.cdf(label[mask == 1]) + self.eps)
        if len(input[mask == -1]) > 0:
            normal_left = torch.distributions.normal.Normal(input[mask == -1], torch.sqrt(var[mask == -1]))
            loss[mask == -1] = - torch.log(normal_left.cdf(label[mask == -1]) + self.eps)
        
        if not torch.isfinite(torch.max(loss)):
            print('Warning found inf in Tobit Loss.')
            print('Observed loss: ', loss[mask == 0])
            print('Right Censored Loss: ', loss[mask == 1])
            print('Left Censored Loss: ', loss[mask == -1])
            
        return torch.mean(loss) if self.reduction == 'mean' else loss


class EvidentialLoss(_Loss):
    '''
    Re-implementation of the Deep Evidential Loss from Amini et al., (2020).
    '''
    def __init__(self, reduction='mean', coeff=1.0, omega=0.01, kl=False) -> None:
        super().__init__()
        assert reduction in ['none', 'mean'], 'Reduce in censored loss should be none or mean'
        self.reduction = reduction
        self.coeff = coeff
        self.omega = omega
        self.kl = kl
        
    def NIG_NLL(self, target, gamma, v, alpha, beta):
        twoBlambda = 2*beta*(1+v)

        nll = 0.5*torch.log(np.pi/v)  \
            - alpha*torch.log(twoBlambda)  \
            + (alpha+0.5) * torch.log(v*(target-gamma)**2 + twoBlambda)  \
            + torch.lgamma(alpha)  \
            - torch.lgamma(alpha+0.5)

        return torch.mean(nll) if self.reduction == 'mean' else nll

    def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
        KL = 0.5*(a1-1)/b1 * (v2*torch.sqrt(mu2-mu1))  \
            + 0.5*v2/v1  \
            - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
            - 0.5 + a2*torch.log(b1/b2)  \
            - (torch.lgamma(a1) - torch.lgamma(a2))  \
            + (a1 - a2)*torch.digamma(a1)  \
            - (b1 - b2)*a1/b1
        return KL

    def NIG_Reg(self, target, gamma, v, alpha, beta):
        error = torch.abs(target-gamma)

        if self.kl: 
            kl = self.KL_NIG(gamma, v, alpha, beta, gamma, self.omega, 1+self.omega, beta)
            reg = error*kl
        else:
            evi = 2*v+(alpha)
            reg = error*evi

        return torch.mean(reg) if self.reduction == 'mean' else reg
        
    def forward(self, target, gamma, v, alpha, beta):
        loss_nll = self.NIG_NLL(target, gamma, v, alpha, beta)
        loss_reg = self.NIG_Reg(target, gamma, v, alpha, beta)
        return loss_nll + self.coeff * loss_reg

class CensoredEvidentialLoss(_Loss):
    '''
    Extension of the Deep Evidential Loss from Amini et al., (2020) that handles censored labels.
    '''
    def __init__(self, reduction='mean', coeff=1.0, omega=0.01, kl=False) -> None:
        super().__init__()
        assert reduction in ['none', 'mean'], 'Reduce in censored loss should be none or mean'
        self.reduction = reduction
        self.coeff = coeff
        self.omega = omega
        self.kl = kl
        
    def NIG_NLL(self, target, gamma, v, alpha, beta):
        twoBlambda = 2*beta*(1+v)

        nll = 0.5*torch.log(np.pi/v)  \
            - alpha*torch.log(twoBlambda)  \
            + (alpha+0.5) * torch.log(v*(target-gamma)**2 + twoBlambda)  \
            + torch.lgamma(alpha)  \
            - torch.lgamma(alpha+0.5)

        return nll

    def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
        KL = 0.5*(a1-1)/b1 * (v2*torch.sqrt(mu2-mu1))  \
            + 0.5*v2/v1  \
            - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
            - 0.5 + a2*torch.log(b1/b2)  \
            - (torch.lgamma(a1) - torch.lgamma(a2))  \
            + (a1 - a2)*torch.digamma(a1)  \
            - (b1 - b2)*a1/b1
        return KL

    def NIG_Reg(self, target, gamma, v, alpha, beta):
        error = torch.abs(target-gamma)

        if self.kl: 
            kl = self.KL_NIG(gamma, v, alpha, beta, gamma, self.omega, 1+self.omega, beta)
            reg = error*kl
        else:
            evi = 2*v+(alpha)
            reg = error*evi

        return reg
        
    def forward(self, target, gamma, v, alpha, beta):
        mask = target['Operator']
        target = target['Label']
        
        diff = target - gamma
        diff[mask == 1] = torch.clamp(diff[mask == 1], min=0.0)
        diff[mask == -1] = torch.clamp(diff[mask == -1], max=0.0)
        diff = diff ** 2
        correct = torch.logical_and(mask != 0, diff == 0)
        
        loss_nll = self.NIG_NLL(target, gamma, v, alpha, beta)
        loss_nll[correct] = 0
        loss_nll = torch.mean(loss_nll) if self.reduction == 'mean' else loss_nll
        
        loss_reg = self.NIG_Reg(target, gamma, v, alpha, beta)
        loss_reg[correct] = 0
        loss_reg = torch.mean(loss_reg) if self.reduction == 'mean' else loss_reg

        return loss_nll + self.coeff * loss_reg


class BayesLoss(_Loss):
    '''
    Re-implementation of the Bayes by Backprop Loss from Blundell et al., (2015).
    '''
    def __init__(self, likelihood, n_train_batches) -> None:
        super().__init__()
        self.likelihood = likelihood
        self.n_train_batches = n_train_batches

    def forward(self, input, target, kl):
        log_likelihood = self.likelihood(input, target).sum()
        if not isinstance(self.likelihood, torch.nn.BCEWithLogitsLoss):
            log_likelihood /= 2
        kl /= self.n_train_batches
        return kl + log_likelihood
    

class ExtendedCensoredEvidentialLoss(_Loss):
    """
    An extended censored evidential loss:
    Uncensored - use student-t PDF (negative log-likelihood)
    Left-censored - use -log CDF
    Right censored - use -log(1-CDF)

    NIG parameters: (gamma, v, alpha, beta)
    Predictive distribution is Student-t with:
        df = 2 * alpha
        loc = gamma
        scale = sqrt(beta * (1 + v) / (v * alpha))
    """

    def __init__(self, reduction='mean', coeff=1.0, omega=0.01, kl=False, eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.coeff = coeff
        self.omega = omega
        self.kl = kl
        self.eps = eps
    
    def student_t_params(self, gamma, v, alpha, beta):
        df = 2 * alpha
        scale = torch.sqrt(beta * (1 + v) / (v * alpha))
        return df, gamma, scale
    
    def NIG_Reg(self, target, gamma, v, alpha, beta):
        error = torch.abs(target - gamma)
        evidence = 2 * v + alpha   # standard evidential regularizer
        return error * evidence
    
    def forward(self, target_dict, gamma, v, alpha, beta):
        mask = target_dict["Operator"]      # -1 = left, 0 = uncensored, +1 = right
        y = target_dict["Label"]

        df, loc, scale = self.student_t_params(gamma, v, alpha, beta)
        dist = StudentT(df=df, loc=loc, scale=scale)

        loss = torch.zeros_like(y)

        # For uncensored points, use the negative log likelihood
        unc_mask = (mask == 0)
        if unc_mask.any():
            loss[unc_mask] = -dist.log_prob(y[unc_mask])

        # For right censored points, use - log(1-CDF)
        right_mask = (mask == 1)
        if right_mask.any():
            tail_prob = 1.0 - dist.cdf(y[right_mask])
            loss[right_mask] = -torch.log(tail_prob + self.eps)

        # For left censored points, use -log(CDF)
        left_mask = (mask == -1)
        if left_mask.any():
            cdf_val = dist.cdf(y[left_mask])
            loss[left_mask] = -torch.log(cdf_val + self.eps)

        # evidential regularizer (only uncensored contributes fully, censored points work partially)
        reg = self.NIG_Reg(y, gamma, v, alpha, beta)
        loss = loss + self.coeff * reg

        return loss.mean() if self.reduction == 'mean' else loss