from ddp_runtime_policy import resolve_ddp_static_graph


def test_static_graph_is_disabled_by_default():
    env = {}
    decision = resolve_ddp_static_graph(env)
    assert decision.requested is False
    assert decision.enabled is False
    assert env["DDP_STATIC_GRAPH"] == "0"


def test_current_optimized_schedule_disables_requested_static_graph():
    env = {
        "DDP_STATIC_GRAPH": "1",
        "GRADIENT_ACCUMULATION_STEPS": "2",
        "USE_LOCAL_HARD_NEGATIVES": "1",
        "LOCAL_HARD_NEGATIVE_WEIGHT": "0.25",
        "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES": "2",
        "USE_IMAGE_PAIR_CONTRASTIVE": "1",
        "IMAGE_PAIR_LOSS_WEIGHT": "0.40",
        "SPAN_DTW_BACKEND": "jax",
    }
    decision = resolve_ddp_static_graph(env)
    assert decision.requested is True
    assert decision.enabled is False
    assert env["DDP_STATIC_GRAPH"] == "0"
    assert "gradient accumulation" in decision.description
    assert "local hard-negative" in decision.description
    assert "Torch-to-JAX" in decision.description


def test_static_graph_can_be_enabled_for_a_truly_fixed_graph():
    env = {
        "DDP_STATIC_GRAPH": "1",
        "GRADIENT_ACCUMULATION_STEPS": "1",
        "USE_LOCAL_HARD_NEGATIVES": "0",
        "USE_IMAGE_PAIR_CONTRASTIVE": "0",
        "SPAN_DTW_BACKEND": "torch",
    }
    decision = resolve_ddp_static_graph(env)
    assert decision.enabled is True
    assert env["DDP_STATIC_GRAPH"] == "1"


def test_force_flag_is_explicitly_recorded():
    env = {
        "DDP_STATIC_GRAPH": "1",
        "FORCE_DDP_STATIC_GRAPH": "1",
        "GRADIENT_ACCUMULATION_STEPS": "2",
        "USE_LOCAL_HARD_NEGATIVES": "1",
        "USE_IMAGE_PAIR_CONTRASTIVE": "1",
        "SPAN_DTW_BACKEND": "jax",
    }
    decision = resolve_ddp_static_graph(env)
    assert decision.forced is True
    assert decision.enabled is True
