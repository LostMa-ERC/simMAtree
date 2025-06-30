import torch


def expected_yule_tree_size(LDA, lda, gamma, mu, Nact, Ninact=0):
    expected_size_act = (1 + (LDA / (lda + gamma - mu))) * torch.exp(
        (lda + gamma - mu) * Nact
    ) - (LDA / (lda + gamma - mu))
    expected_loss_inact = torch.exp(-mu * Ninact)
    return expected_size_act * expected_loss_inact
