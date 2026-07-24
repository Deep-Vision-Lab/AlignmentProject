from types import SimpleNamespace

import torch

from training_optimizations import optimized_train_one_epoch


class Context:
    world_size = 1
    is_main = False


class FakeTrainModule:
    P = SimpleNamespace(log_memory_every_n_batches=0)
    CTX = Context()
    USE_AMP = False

    @staticmethod
    def has_trainable_parameters(module):
        return any(parameter.requires_grad for parameter in module.parameters())

    @staticmethod
    def _batch_size(batch):
        return int(batch[0].shape[0])

    @staticmethod
    def _accumulate_stats(target, stats, weight):
        for key, value in stats.items():
            target[key] = target.get(key, 0.0) + float(value) * weight

    @staticmethod
    def _merge_epoch_payload(payload):
        weight = max(1, payload["weight"])
        return (
            payload["loss_sum"] / weight,
            {key: value / weight for key, value in payload["stats_sum"].items()},
        )

    @staticmethod
    def _format_memory(_text_encoder):
        return ""

    @staticmethod
    def compute_batch_loss(model, _text_encoder, _criterion, batch):
        inputs, targets = batch
        prediction = model(inputs)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        return loss, {"total": float(loss.detach())}


def _loader():
    return [
        (torch.tensor([[1.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[2.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[3.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[4.0]]), torch.tensor([[0.0]])),
    ]


def _manual(initial_weight):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(initial_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = _loader()
    for start in range(0, len(batches), 2):
        optimizer.zero_grad(set_to_none=True)
        for inputs, targets in batches[start : start + 2]:
            loss = torch.nn.functional.mse_loss(model(inputs), targets) / 2
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
        optimizer.step()
    return model.weight.detach().clone()


def test_accumulation_matches_manual_microbatch_average(monkeypatch):
    monkeypatch.setenv("GRADIENT_ACCUMULATION_STEPS", "2")
    monkeypatch.delenv("PROFILE_MAX_BATCHES", raising=False)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.5)
    text_encoder = torch.nn.Identity()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    train_epoch = optimized_train_one_epoch(FakeTrainModule)
    train_epoch(model, text_encoder, None, optimizer, scaler, _loader())

    torch.testing.assert_close(model.weight, _manual(0.5), rtol=1e-6, atol=1e-6)
