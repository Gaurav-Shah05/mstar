

def test_pi05_adarms_norm_with_precomputed_modulation_matches_adarms_norm():
    """Feeding a norm its row of the precomputed modulations is the stock
    AdaRMSNorm on the condition, for every request in the batch."""
    from mstar.model.components import AdaRMSNorm
    from mstar.model.pi05.components.action_expert import Pi05AdaRMSNorm

    torch.manual_seed(0)
    hidden, rows_per_request, requests = 16, 3, 2
    stock = AdaRMSNorm(hidden, cond_dim=hidden)
    torch.nn.init.normal_(stock.dense.weight)
    torch.nn.init.normal_(stock.dense.bias)
    ours = Pi05AdaRMSNorm(hidden, cond_dim=hidden, eps=stock.variance_epsilon, slot=1)
    ours.load_state_dict(stock.state_dict())

    x = torch.randn(requests * rows_per_request, hidden)
    cond = torch.randn(1, hidden)
    other_norm_row = torch.randn(1, 3 * hidden)
    modulations = torch.cat([other_norm_row, ours.dense(cond)])  # this norm is slot 1

    want_normed, want_gate = stock(x, cond.expand(requests, -1))
    got_normed, got_gate = ours(x, modulations)
    torch.testing.assert_close(got_normed, want_normed)
    torch.testing.assert_close(got_gate, want_gate)
