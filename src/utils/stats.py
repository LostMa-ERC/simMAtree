import torch


def expected_yule_tree_size(LDA, lda, gamma, mu, Nact, Ninact=0):
    return Nact * LDA * torch.exp((lda + gamma - mu) * Nact - mu * Ninact)
