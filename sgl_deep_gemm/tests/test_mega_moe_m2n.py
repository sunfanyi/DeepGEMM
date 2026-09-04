"""M2N correctness, without benchmarking.

Single GPU: python test_mega_moe_m2n.py
AG2/EG2: torchrun --standalone --nproc_per_node=4 test_mega_moe_m2n.py
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

import deep_gemm
from deep_gemm import _C
from deep_gemm.mega.m2n import (
    get_symm_buffer_for_mega_moe_m2n,
    make_m2n_weight_descs,
    mega_moe_m2n_ag,
    mega_moe_m2n_eg,
)
from deep_gemm.utils import (
    cast_back_from_fp4,
    per_token_cast_to_fp4,
    per_token_cast_to_fp8,
)

HIDDEN, INTERMEDIATE, EXPERTS, TOPK, CAPACITY = 512, 512, 8, 2, 128
CLAMP = 1.25
OUTPUT_SCALES = (1.0, 2.5)


def test_layout() -> None:
    ag_size, eg_size = 2, 2
    num_experts, capacity, num_topk = 8, 128, 2
    hidden, intermediate_hidden = 512, 512
    num_bytes, pool_tokens, slice_buffers = _C.get_symm_buffer_size_for_mega_moe_m2n(
        ag_size,
        eg_size,
        num_experts,
        capacity,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    buffer = torch.zeros(int(num_bytes), dtype=torch.int8, device="cuda")
    _C.initialize_symm_buffer_for_mega_moe_m2n(buffer, ag_size + eg_size)
    views = tuple(map(torch.from_dlpack, slice_buffers(buffer)))
    x, x_sf, topk_ids, topk_weights, l1, l1_sf, l2, l2_sf = views

    assert int(pool_tokens) > 0
    assert x.shape == (capacity, hidden)
    assert x_sf.shape == (capacity, hidden // 128)
    assert topk_ids.shape == (capacity, num_topk)
    assert topk_weights.shape == (capacity, num_topk)
    assert l1.shape == (int(pool_tokens), hidden)
    assert l2.shape == (int(pool_tokens), intermediate_hidden)
    assert l1_sf.shape[1] == hidden // 128
    assert l2_sf.shape[1] == intermediate_hidden // 128
    assert l1_sf.stride() == (1, l1_sf.shape[0])
    assert l2_sf.stride() == (1, l2_sf.shape[0])

    base = buffer.data_ptr()
    end = base + buffer.numel()
    assert all(base <= view.data_ptr() < end for view in views)
    seeded_words = buffer.view(torch.int32)
    assert torch.count_nonzero(seeded_words).item() == 1
    assert seeded_words.max().item() == ag_size + eg_size


def _make_weights(layer: int):
    generator = torch.Generator(device="cuda").manual_seed(100 + layer)
    quantized, reference = [], []
    for n, k in ((2 * INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE)):
        raw = (
            torch.randn(
                (EXPERTS * n, k),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.125
        )
        packed, scales = per_token_cast_to_fp4(raw, use_ue8m0=True, gran_k=32)
        reference.append(
            cast_back_from_fp4(packed, scales, gran_k=32).view(EXPERTS, n, k)
        )
        quantized.append(
            (packed.view(EXPERTS, n, k // 2), scales.view(EXPERTS, n, k // 32))
        )
    return quantized, reference


def _fp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    packed, scales = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
    return (packed.float().view(x.shape[0], -1, 32) * scales.unsqueeze(-1)).view_as(x)


def _reference(x, ids, weights, layer_weights, output_scale):
    # Match the actual quantized operands, not the original BF16 weights.
    x = _fp8_roundtrip(x.float())
    output = torch.zeros_like(x)
    for expert in range(EXPERTS):
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        gate_up = (x[rows] @ layer_weights[0][expert].T).bfloat16().float()
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=CLAMP)
        up = up.clamp(-CLAMP, CLAMP)
        activation = (gate / (1 + torch.exp(-gate))) * up
        activation = _fp8_roundtrip(activation * weights[rows, slots, None])
        expert_output = (activation @ layer_weights[1][expert].T).bfloat16().float()
        output.index_add_(0, rows, expert_output)
    return (output * output_scale).bfloat16()


def _make_input(rank, layer, lane, count, phase):
    generator = torch.Generator(device="cuda").manual_seed(
        1000 * phase + 100 * rank + 10 * layer + lane
    )
    # Mirror SGLang's non-null, unrouted sentinel for an empty A participant.
    x = torch.randn(
        (max(count, 1), HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    rows = torch.arange(x.shape[0], device="cuda")
    first = (rows + 3 * rank + layer + lane) % EXPERTS
    ids = torch.stack((first, (first + EXPERTS // 2) % EXPERTS), dim=1).int()
    weights = (
        torch.rand((x.shape[0], TOPK), generator=generator, device="cuda") * 0.75
        + 0.125
    )
    ids[::3, 1] = -1
    if count == 0:
        x.zero_()
        ids.fill_(-1)
        weights.zero_()
    return x, ids, weights


def _check_output(actual, expected, ids, label):
    assert torch.isfinite(actual).all(), label
    empty_rows = (ids < 0).all(dim=1)
    assert torch.count_nonzero(actual[empty_rows]).item() == 0, label
    rms = expected.float().square().mean().sqrt().item()
    error = actual.float() - expected.float()
    if rms == 0:
        assert (
            empty_rows.all()
        ), f"{label}: active routes produced an all-zero reference"
        assert torch.count_nonzero(actual).item() == 0, label
        return
    relative_rms = error.square().mean().sqrt().item() / rms
    print(f"{label}: relative_rms={relative_rms:.6g}", flush=True)
    # Use a BF16-scale RMS error budget for the independently accumulated reference.
    assert relative_rms <= 2**-8, label


def run_distributed() -> None:
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    assert dist.get_world_size() == 4, "Use torchrun --nproc_per_node=4 (AG2/EG2)"
    rank, group = dist.get_rank(), dist.group.WORLD
    torch.backends.cuda.matmul.allow_tf32 = False
    buffers = [
        get_symm_buffer_for_mega_moe_m2n(
            group,
            ag_size=2,
            eg_size=2,
            num_experts=EXPERTS,
            num_max_tokens_per_rank=CAPACITY,
            num_topk=TOPK,
            hidden=HIDDEN,
            intermediate_hidden=INTERMEDIATE,
        )
        for _ in range(3)
    ]
    layers = [_make_weights(layer) for layer in range(2)]
    weight_storage, descriptors = [], []
    if rank >= 2:
        local_experts = EXPERTS // 2
        start = (rank - 2) * local_experts
        for quantized, _ in layers:
            local_weights = []
            for packed, scales in quantized:
                n, k = packed.shape[1], packed.shape[2] * 2
                packed = packed[start : start + local_experts].contiguous()
                scales = deep_gemm.transform_sf_into_required_layout(
                    scales[start : start + local_experts].contiguous(),
                    n,
                    k,
                    (1, 32),
                    local_experts,
                )
                local_weights.append((packed, scales))
            transformed = deep_gemm.transform_weights_for_mega_moe(*local_weights)
            weight_storage.append(transformed)
            descriptors.append(
                make_m2n_weight_descs(
                    *transformed,
                    num_experts_per_rank=local_experts,
                    hidden=HIDDEN,
                    intermediate_hidden=INTERMEDIATE,
                )
            )
        descriptors = torch.stack(descriptors).cuda()
    streams = [torch.cuda.Stream() for _ in range(3)] if rank < 2 else []

    # Same allocations cross N1/N3 phases, including partial/all-empty lanes
    # and unequal counts. Each phase is one persistent EG for two layers.
    phases = (
        ((32,), (17,)),
        ((17, 7, 3), (0, 11, 1)),
        ((0,), (0,)),
        ((1, 0, 33), (9, 0, 17)),
    )
    for phase, counts in enumerate(phases):
        num_lanes = len(counts[0])
        inputs, outputs = [], []
        if rank < 2:
            inputs = [
                [
                    _make_input(rank, layer, lane, count, phase)
                    for lane, count in enumerate(counts[rank])
                ]
                for layer in range(2)
            ]
            outputs = [
                [torch.full_like(data[0], float("nan")) for data in layer]
                for layer in inputs
            ]
        torch.cuda.synchronize()
        dist.barrier()
        if rank < 2:
            for stream in streams[:num_lanes]:
                stream.wait_stream(torch.cuda.current_stream())
            for layer in range(2):
                for lane in range(num_lanes):
                    with torch.cuda.stream(streams[lane]):
                        mega_moe_m2n_ag(
                            outputs[layer][lane],
                            *inputs[layer][lane],
                            buffers[lane],
                            output_scale=OUTPUT_SCALES[layer],
                            num_sms=24,
                        )
        else:
            mega_moe_m2n_eg(
                descriptors,
                buffers[:num_lanes],
                expected_num_tokens_per_rank=max(1, max(map(max, counts))),
                num_experts_per_rank=EXPERTS // 2,
                use_fp8_weights=False,
                activation_clamp=CLAMP,
                fast_math=True,
            )
        torch.cuda.synchronize()
        dist.barrier()
        if rank < 2:
            for layer in range(2):
                for lane in range(num_lanes):
                    data = inputs[layer][lane]
                    expected = _reference(*data, layers[layer][1], OUTPUT_SCALES[layer])
                    _check_output(
                        outputs[layer][lane],
                        expected,
                        data[1],
                        f"rank={rank} phase={phase} layer={layer} lane={lane}",
                    )
        dist.barrier()

    torch.cuda.synchronize()
    for buffer in buffers:
        buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        test_layout()
    else:
        run_distributed()
    print("OK")
