

def conservative_strategy(obs):
    # obs = [pd, utilization, debt_ratio, overdue_count, inquiries, credit_limit]
    pd_prob = obs[0]
    if pd_prob > 0.4:
        return 0  # 降额
    return 1  # 维持

def rule_based_strategy(obs):
    pd_prob = obs[0]
    utilization = obs[1]

    if pd_prob > 0.5:
        return 0
    elif pd_prob < 0.2 and utilization > 0.6:
        return 2
    else:
        return 1