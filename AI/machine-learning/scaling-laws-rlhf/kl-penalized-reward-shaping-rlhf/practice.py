"""Practice for KL-Penalized Reward Shaping RLHF."""

def categorical_kl(policy_probs, reference_probs):
    raise NotImplementedError

def sampled_kl_from_logprobs(policy_logprobs, reference_logprobs):
    raise NotImplementedError

def shaped_reward(reward_model_score, sampled_kl, beta):
    raise NotImplementedError
