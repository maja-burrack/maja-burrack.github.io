import random

def metropolis_hastings(
    p,
    q,
    initial_state,
    n
):
    current_x = initial_state
    
    chain = [current_x]
    for i in range(n):
        u = random.uniform(0, 1)
        proposal_x = q(current_x)
        A = min(1, (p(proposal_x)/p(current_x))*(q(current_x)/q(proposal_x)))
        if A >= u:
            current_x = proposal_x
        chain.append(current_x)
    
    return chain
