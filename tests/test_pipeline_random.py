"""Random sampling in synchronous (CPU) and asynchronous (CUDA) pipeline speculation."""

import numpy as np
import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.worker.sampler import Sampler

PROMPT = [2, 3, 4]
MODES = ["sync", pytest.param("async", marks=pytest.mark.gpu)]


def make_model(seed=17):
    torch.manual_seed(seed)
    return OuroForCausalLM(tiny_ouro_config())


def make_engine(model, mode, max_num_seqs=16):
    speculative_config = SpeculativeConfig(3, draft_loops=2, mode="pipeline")
    if mode == "sync":
        return LLMEngine(
            model,
            speculative_config=speculative_config,
            cache_config=CacheConfig(256, 2),
            scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=64),
        )
    return StageEngine(
        model.cuda(),
        attention_backend="triton",
        cache_config=CacheConfig(256, 2, enable_prefix_caching=False, incremental_allocation=True),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=64),
        exit_config=ExitConfig("ouro"),
        execution_config=ExecutionConfig(async_scheduling=True, multi_stream=True),
        speculative_config=speculative_config,
    )


def drain(engine):
    outputs = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            outputs[out.request_id] = out
    return outputs


def target_distribution(model, sampling):
    """Exact p(t2 | prompt, t1) for the most likely t1, computed without speculation."""
    cache = LLMEngine(model, cache_config=CacheConfig(128, 2)).cache_manager
    with torch.inference_mode():
        assert cache.allocate("ref", 8)
        h = model.prelude(torch.tensor(PROMPT))
        for d in range(4):
            h, _ = model.recurrent(h, ["ref"] * 3, [d] * 3, [0, 1, 2], cache)
        t1 = int(Sampler.probabilities(model.coda(h[-1:]).squeeze(0), sampling).argmax())
        h = model.prelude(torch.tensor([t1]))
        for d in range(4):
            h, _ = model.recurrent(h, ["ref"], [d], [3], cache)
        return t1, Sampler.probabilities(model.coda(h).squeeze(0), sampling)


@pytest.mark.parametrize("mode", MODES)
def test_pipeline_random_matches_target_distribution(mode):
    """The second token goes through draft + rejection sampling; it must follow exact p."""
    model = make_model()
    t1, p = target_distribution(model, SamplingParams(temperature=1.2, top_k=6))
    support = torch.nonzero(p > 0).squeeze(-1).tolist()
    engine = make_engine(model, mode)
    counts = dict.fromkeys(support, 0)
    for start in range(0, 1600, 16):
        for seed in range(start + 1, start + 17):
            params = SamplingParams(
                max_tokens=2, temperature=1.2, top_k=6, seed=seed, ignore_eos=True
            )
            engine.add_request(f"s{seed}", PROMPT, params)
        for out in drain(engine).values():
            if out.token_ids[0] == t1:
                counts[out.token_ids[1]] += 1  # KeyError means an out-of-support token
    total = sum(counts.values())
    assert total >= 150
    observed = np.array([counts[t] for t in support], dtype=np.float64)
    expected = p[support].double().cpu().numpy() * total
    # 99.9% chi-square critical value for 5 degrees of freedom.
    assert len(support) == 6
    assert ((observed - expected) ** 2 / expected).sum() < 20.515


@pytest.mark.parametrize("mode", MODES)
def test_pipeline_random_keeps_greedy_neighbours_and_replays_seed(mode):
    model = make_model(seed=42)
    greedy = SamplingParams(max_tokens=8, ignore_eos=True)

    def run(seed):
        engine = make_engine(model, mode, max_num_seqs=4)
        engine.add_request("g", PROMPT, greedy)
        if seed is not None:
            params = SamplingParams(
                max_tokens=8, temperature=1.0, top_k=16, top_p=0.9, seed=seed, ignore_eos=True
            )
            engine.add_request("r", [5, 6], params)
        outputs = drain(engine)
        assert engine.cache_manager.num_used_blocks == 0 and not engine.rng.generators
        return {rid: out.token_ids for rid, out in outputs.items()}

    alone, first, replay, other = run(None), run(7), run(7), run(8)
    assert first["g"] == alone["g"] == other["g"]
    assert first["r"] == replay["r"] != other["r"]


def test_pipeline_random_rejects_unsupported_combinations():
    model = make_model()
    model.requires_ordered_pipeline_coda = True
    coda = StageEngine(model, speculative_config=SpeculativeConfig(3, mode="pipeline"))
    preemption = StageEngine(
        make_model(),
        speculative_config=SpeculativeConfig(3, mode="pipeline"),
        scheduler_config=SchedulerConfig(enable_preemption=True),
    )
    for engine in (coda, preemption):
        with pytest.raises(ValueError, match="greedy"):
            engine.add_request("r", PROMPT, SamplingParams(max_tokens=4, temperature=0.8))
