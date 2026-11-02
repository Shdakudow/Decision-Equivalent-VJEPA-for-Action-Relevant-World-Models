"""Check the released loss paths without training or changing parameters.

Uses synthetic tensors and the original model/loss functions. Also checks
the included release-file hashes. Run from the repository root with PyTorch, NumPy,
PyYAML, and tqdm available. No data, checkpoints, or GPU are required.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from devjepa.losses import predictor_loss
from devjepa.models import GaussianPolicy, VariationalPredictor, posterior_kl


def main():
    with (ROOT / 'paper/released_implementation_checksums.csv').open() as handle:
        files = list(csv.DictReader(handle))
    for row in files:
        path = ROOT / row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = VariationalPredictor(512, 2, 1, 1024, 64, 32, 1)
    policy = GaussianPolicy(512, 2, 1, 512, 32).eval().requires_grad_(False)
    z, target = torch.randn(2, 512), torch.randn(2, 1, 512)
    actions, tasks = torch.randn(2, 1, 2), torch.zeros(2, dtype=torch.long)
    stochastic_inputs = []
    hook = model.predictor.register_forward_pre_hook(
        lambda module, inputs: stochastic_inputs.append(inputs[0][:, -64:].detach())
    )
    posterior = model(z, actions, tasks, target, sample=True)
    prior = model(z, actions, tasks, target=None, sample=False)
    hook.remove()
    assert torch.count_nonzero(stochastic_inputs[-1]) == 0
    loss, _ = predictor_loss(
        'hybrid', prior.predicted, target, tasks,
        prior.posterior_mean, prior.posterior_logvar, policy,
        beta=0., epsilon0=.01, uncertainty_scale=.05, gamma=.95,
        decision_weight=.01,
    )
    posterior_parameters = list(model.posterior.parameters())
    gradients = torch.autograd.grad(loss, posterior_parameters,
                                    allow_unused=True, retain_graph=True)
    assert all(g is None for g in gradients)
    decoder_gradients = torch.autograd.grad(loss, list(model.predictor.parameters()),
                                            retain_graph=True)
    assert any(torch.count_nonzero(g) for g in decoder_gradients)
    regularizer = posterior_kl(posterior.posterior_mean, posterior.posterior_logvar).mean()
    posterior_gradients = torch.autograd.grad(regularizer, posterior_parameters)
    assert any(torch.count_nonzero(g) for g in posterior_gradients)
    print(json.dumps({
        'release_file_checksums_verified': len(files),
        'prediction_uses_h_zero': True,
        'prediction_and_policy_loss_reaches_posterior_network': False,
        'prediction_and_policy_loss_reaches_point_predictor': True,
        'posterior_KL_reaches_posterior_network': True,
        'optimizer_steps': 0,
        'scope': 'Source dependency check only; not a reproduction of historical experiments.',
    }, indent=2))


if __name__ == '__main__':
    main()
