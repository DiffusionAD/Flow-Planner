import torch

def traj_chunking(future, action_length, action_overlap):
    delta = action_length - action_overlap
    index = delta
    actions = []
    while index + action_overlap <= future.shape[-2]:
        action = future[..., index - delta:index - delta + action_length, :]
        actions.append(action)
        index += delta

    return actions

def average_assemble(x, future_length, action_length, action_overlap, state_dim):
    B = x.shape[0]
    final_action = torch.zeros((B, 1, future_length, state_dim), device=x.device)
    pos_cnt = torch.zeros((1, 1, future_length, 1), device=x.device)
    for i in range(0, x.shape[1]):
        start_pivot = i*(action_length - action_overlap)
        final_action[:, :, start_pivot:start_pivot+action_length, :] += x[:, i:i+1, :].reshape(B, -1, action_length, state_dim)
        pos_cnt[:, :, start_pivot:start_pivot+action_length, :] += 1

    return final_action / pos_cnt

def linear_assemble(x, future_length, action_length, action_overlap, state_dim):
    '''
    x: (B, N, L * D)
    Linear weighting combination of overlapping segment
    '''
    B = x.shape[0]
    final_action = torch.zeros((B, 1, future_length, state_dim), device=x.device)
    
    weights = torch.linspace(0, 1, action_overlap)
    reverse_weights = torch.linspace(1, 0, action_overlap)
    first_weights = torch.ones((1, 1, action_length, 1), device=x.device)
    first_weights[0, 0, -action_overlap:, 0] = reverse_weights
    last_weights = torch.ones((1, 1, action_length, 1), device=x.device)
    last_weights[0, 0, :action_overlap, 0] = weights
    mid_weights = torch.ones((1, 1, action_length, 1), device=x.device)
    mid_weights[0, 0, -action_overlap:, 0] = reverse_weights
    mid_weights[0, 0, :action_overlap, 0] = weights
    for i in range(0, x.shape[1]):
        start_pivot = i*(action_length - action_overlap)
        if i == 0: 
            final_action[:, :, start_pivot:start_pivot+action_length, :] += x[:, i:i+1, :].reshape(B, -1, action_length, state_dim) * first_weights
        elif i == x.shape[1] - 1:
            final_action[:, :, start_pivot:start_pivot+action_length, :] += x[:, i:i+1, :].reshape(B, -1, action_length, state_dim) * last_weights
        else:
            final_action[:, :, start_pivot:start_pivot+action_length, :] += x[:, i:i+1, :].reshape(B, -1, action_length, state_dim) * mid_weights
    
    return final_action
    
    
def assemble_actions(x, future_length, action_length, action_overlap, state_dim, method='average'):
    '''
    assemble the actions with overlap into one complete trajectory
    :params
        x: (B, P, action_length * state_dim), where P is the number of actions
    '''
    if method == 'average':
        return average_assemble(x, future_length, action_length, action_overlap, state_dim)
    if method == 'linear':
        assert action_length >= 2 * action_overlap, f"linear smoothening is not supported for tokens with overlap > length / 2"
        return linear_assemble(x, future_length, action_length, action_overlap, state_dim)