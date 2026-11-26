
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

def parallelize_qwen(model, tp_mesh):
    if tp_mesh.size() > 1:
        for i, block in enumerate(model.model.language_model.layers):
            plan = {
                "self_attn.q_proj":  PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                "self_attn.k_proj":  PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                "self_attn.v_proj":  PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                 "self_attn.o_proj": PrepareModuleInput(
                    input_layouts=(Shard(-1)),
                    desired_input_layouts=(Shard(0)),
                    use_local_output=True,
                ),
            }

            self_attn = block.self_attn
            self_attn.num_heads = self_attn.num_heads // tp_mesh.size()
            self_attn.num_key_value_heads = self_attn.num_key_value_heads // tp_mesh.size()
            parallelize_module(block, tp_mesh, plan)

            if i == 0:
                parallelize_module(
                    block,
                    tp_mesh,
                    PrepareModuleInput(
                        input_layouts=(Replicate(), None, None, None),
                        desired_input_layouts=(Shard(0), None, None, None),
                        use_local_output=True,
                    ),
                )
            if i == len(model.model.language_model.layers)-1:
                parallelize_module(
                    block,
                    tp_mesh,
                    PrepareModuleOutput(
                        output_layouts=(Shard(0)),
                        desired_output_layouts=(Replicate())
                    ),
                )

        for i, block in enumerate( model.model.visual.blocks):
            plan = {
                "attn.to_query":  PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                "attn.to_key":  PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                "attn.to_value":  PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                 "attn.proj": PrepareModuleInput(
                    input_layouts=(Shard(-1)),
                    desired_input_layouts=(Shard(0)),
                    use_local_output=True,
                ),
            }
            attn = block.attn
            attn.num_heads = attn.num_heads // tp_mesh.size()
            parallelize_module(block, tp_mesh, plan)

            if i == 0:
                parallelize_module(
                    block,
                    tp_mesh,
                    PrepareModuleInput(
                        input_layouts=(Replicate(), None, Replicate(), Replicate()),
                        desired_input_layouts=(Shard(0), None, Replicate(), Replicate()),
                        use_local_output=True,
                    ),
                )
            if i == len(model.model.visual.blocks)-1:
                parallelize_module(
                    block,
                    tp_mesh,
                    PrepareModuleOutput(
                        output_layouts=(Shard(0)),
                        desired_output_layouts=(Replicate())
                    ),
                )

    return model
