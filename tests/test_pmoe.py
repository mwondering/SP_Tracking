from __future__ import annotations

import torch

from sp_tracking.tasks.tracking.rl.pmoe import (
  OnlineKMeansRouter,
  PeriodicAutoencoder,
  PrototypeRoutedResidualMoE,
)


def test_periodic_autoencoder_reconstructs_and_exposes_phase_invariant_features() -> None:
  torch.manual_seed(3)
  pae = PeriodicAutoencoder(
    input_dim=5,
    latent_dim=3,
    window_length=20,
    window_seconds=0.4,
    hidden_dims=(8,),
    kernel_size=3,
  )
  value = torch.randn(4, 5, 20)

  reconstruction, parameters = pae(value)
  embedding = pae.cluster_embedding(value)

  assert reconstruction.shape == value.shape
  assert embedding.shape == (4, 9)
  assert set(parameters) == {
    "latent",
    "amplitude",
    "frequency",
    "offset",
    "phase_shift",
  }
  assert torch.isfinite(reconstruction).all()
  assert torch.isfinite(embedding).all()


def test_online_kmeans_router_updates_buffers_without_trainable_parameters() -> None:
  router = OnlineKMeansRouter(
    feature_dim=2,
    num_clusters=2,
    temperature=0.2,
    momentum=0.0,
  )
  features = torch.tensor(
    [
      [-2.0, -1.9],
      [-1.8, -2.1],
      [2.0, 1.9],
      [1.8, 2.1],
    ]
  )
  router.update_feature_statistics(
    features.sum(dim=0),
    features.square().sum(dim=0),
    float(features.shape[0]),
  )
  router.initialize(features)
  assignments = router.hard_assignments(features)
  cluster_sums = torch.zeros(2, 2)
  cluster_sums.index_add_(0, assignments, features)
  cluster_counts = torch.bincount(assignments, minlength=2).float()
  router.update_centroids(cluster_sums, cluster_counts)
  probabilities = router.probabilities(features)

  assert not tuple(router.parameters())
  assert bool(router.initialized.item())
  assert int(router.num_updates.item()) == 1
  assert torch.all(cluster_counts > 0)
  torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4))
  assert torch.equal(probabilities.argmax(dim=-1), assignments)


def test_prototype_routed_moe_detaches_routes_but_trains_policy_experts() -> None:
  torch.manual_seed(5)
  moe = PrototypeRoutedResidualMoE(
    input_dim=7,
    output_dim=3,
    context_hidden_dim=12,
    hidden_dim=8,
    num_experts=4,
    top_k=2,
    expansion=2,
  )
  value = torch.randn(6, 7)
  routes = torch.softmax(torch.randn(6, 4), dim=-1).requires_grad_()

  moe(value, routes).sum().backward()

  assert routes.grad is None
  assert any(parameter.grad is not None for parameter in moe.parameters())
  sparse = moe.sparse_probabilities(routes.detach())
  assert torch.equal(
    torch.count_nonzero(sparse, dim=-1),
    torch.full((6,), 2),
  )
  torch.testing.assert_close(sparse.sum(dim=-1), torch.ones(6))
