# import pytest
import torch
from supar.models import POSModel


if __name__ == "__main__":
    model = POSModel(3, 2)
    print(model)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    emit_probs, trans_probs = model(torch.tensor([[2, 0, 2], [2, 0, 2]]), mask)
    print(emit_probs.exp())

    # alpha, logZ = model._forward(emit_probs, mask)
    # print(logZ.exp())
    # print(alpha.exp())

    # beta, logZ  = model._forward(emit_probs, mask, forward=False)
    # print(logZ.exp())
    # print(beta.exp())
    print('------------------')
    # gamma = (alpha + beta).logsumexp(-1)

    gamma, xi = model._e(emit_probs, trans_probs, mask)
    print(f"gamma: {gamma}")

    # batch_size, seq_len, n_cpos = alpha.shape

    # [batch_size, seq_len, n_cpos]
    # [batch_size, seq_len, n_cpos_t, 1] +  + [batch_size, seq_len, 1, n_cpos_t+1] + [batch_size, seq_len, 1, n_cpos_t+1]
    # xi = (alpha[:, :-1].unsqueeze(-1) + trans_probs.reshape(1, 1, n_cpos, n_cpos) + emit_probs[:, 1:].unsqueeze(-2) + beta[:, 1:].unsqueeze(-2)).contiguous().view(batch_size, seq_len-1, -1).logsumexp(-1)
    print(f"xi:    {xi}")
    
