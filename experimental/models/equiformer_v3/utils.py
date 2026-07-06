import torch


def reduce_edge(inputs, edge_index, output_shape):
    # triton 的 atomic_add 无 bf16 路径：bf16 的 index_add 会反融合成慢 aten 核。
    # fp32 累加保持融合的 triton scatter，且数值更稳。fp32/tf32 下为 no-op。
    work_dtype = inputs.dtype
    outputs = torch.zeros(
        *output_shape,
        device=inputs.device,
        dtype=torch.float32,
    )
    outputs.index_add_(0, edge_index, inputs.float())
    return outputs.to(work_dtype)