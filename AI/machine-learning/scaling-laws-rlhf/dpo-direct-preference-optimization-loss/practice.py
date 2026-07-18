"""Practice for DPO Direct Preference Optimization Loss."""

def sigmoid(x):
    raise NotImplementedError

def dpo_margin(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta):
    raise NotImplementedError

def dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta):
    raise NotImplementedError
