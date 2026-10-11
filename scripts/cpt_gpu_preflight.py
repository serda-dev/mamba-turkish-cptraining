"""Exercise the cached Mamba decode kernel before expensive model evaluation."""
import math
import os
import shutil


def check_gpu_runtime():
    compiler = shutil.which(os.environ.get('CC','gcc'))
    if not compiler:
        raise RuntimeError('C compiler required by Triton is missing')
    import torch
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    state = torch.zeros((1,8,16),device='cuda')
    x = torch.ones((1,8),device='cuda')
    dt = torch.ones_like(x)
    a = -torch.ones((8,16),device='cuda')
    b = torch.ones((1,16),device='cuda')
    c = torch.ones_like(b)
    # Jamba supplies both D and dt_bias; this pinned Mamba release requires
    # their stride arguments even though its public signature defaults to None.
    d = torch.zeros(8,device='cuda')
    dt_bias = torch.zeros_like(d)
    first = selective_state_update(state,x,dt,a,b,c,D=d,dt_bias=dt_bias)
    torch.testing.assert_close(first,torch.full_like(x,16.0))
    second = selective_state_update(state,x,dt,a,b,c,D=d,dt_bias=dt_bias)
    torch.testing.assert_close(second,torch.full_like(x,16*(1+math.exp(-1))),rtol=1e-4,atol=1e-4)
    torch.cuda.synchronize()
    return {'gpu':torch.cuda.get_device_name(),'compiler':compiler,'mamba_decode_verified':True}


if __name__ == '__main__':
    import json
    print(json.dumps(check_gpu_runtime()))
