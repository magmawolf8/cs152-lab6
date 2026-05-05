import os
import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "

"""
Performs a 2D convolution operation using NKI.
Args:
    X: Input tensor of shape (batch_size, in_channels, input_height, input_width).
    W: Weight tensor of shape (out_channels, in_channels, filter_height, filter_width).
    bias: Bias tensor of shape (out_channels).
Returns:
    out_tensor: The result of the 2D convolution operation, with shape 
                (batch_size, out_channels, output_height, output_width).
Note:
    For ease of implementation, you can expect the inputs to abide by the following restrictions
    - filter_height == filter_width
    - input_channels % 128 == 0
    - output_channels % 128 == 0
    - output_width * output_height % 512 == 0
"""
@nki.jit
def conv2d_nki(X, W, bias):
    N, C, IH, IW = X.shape
    K, C_, A, B = W.shape
    K_ = bias.shape[0]

    OH = IH - A + 1
    OW = IW - B + 1

    assert A == B, "Filter height must be equal to filter width"
    assert C % 128 == 0, "Input channels must be divisible by 128"
    assert K % 128 == 0, "Output channels must be divisible by 128"
    assert OW * OH % 512 == 0, "Output width * output height must be divisible by 512"

    out = nl.ndarray(
        shape=(N, K, OH, OW),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Tile over output spatial dims (OH, OW). Target TH=16, TW=32
    OH_f2 = 0
    t = OH
    while t % 2 == 0:
        OH_f2 += 1
        t //= 2
    OW_f2 = 0
    t = OW
    while t % 2 == 0:
        OW_f2 += 1
        t //= 2
    TH = 2 ** min(max(4, 9 - OW_f2), OH_f2)
    TW = 2 ** min(max(5, 9 - OH_f2), OW_f2)
    TC = nl.tile_size.pmax
    TK = nl.tile_size.gemm_stationary_fmax

    for n in nl.affine_range(N):
        for th in nl.affine_range(OH // TH):
            for tw in nl.affine_range(OW // TW):
                for tk in nl.affine_range(K // TK):
                    # input halo ping pong buffers
                    halo_buf = [
                        nl.ndarray(shape=(TC, TH + A - 1, TW + B - 1), dtype=X.dtype, buffer=nl.sbuf),
                        nl.ndarray(shape=(TC, TH + A - 1, TW + B - 1), dtype=X.dtype, buffer=nl.sbuf),
                    ]
            
                    # psum buffer
                    psum_tile = nl.zeros(shape=(TK, TH * TW), dtype=nl.float32, buffer=nl.psum)

                    # packed filter ping pong buffers (ready for GEMM)
                    w_tile = [
                        [nl.ndarray(shape=(TC, TK), dtype=X.dtype, buffer=nl.sbuf) for _ in range(B)]
                        for _ in range(A)
                    ]

                    # raw filter weights, (TK, TC, A, B), TK on P
                    w_raw = nl.ndarray(shape=(TK, TC, A, B), dtype=X.dtype, buffer=nl.sbuf)


                    # prime
                    nisa.dma_copy(dst=halo_buf[0], src=X[n, 0:TC, 
                                                th * TH : th * TH + TH + A - 1, 
                                                tw * TW : tw * TW + TW + B - 1])
                    for tc in nl.static_range(C // TC):
                        # prefetch
                        if tc < (C // TC) - 1:
                            nisa.dma_copy(dst=halo_buf[(tc + 1) % 2], src=X[n, (tc + 1) * TC:(tc + 2) * TC, th * TH : th * TH + TH + A - 1, tw * TW : tw * TW + TW + B - 1])
                        
                        # load filter
                        w_raw[...] = nl.load(W[tk * TK:(tk + 1) * TK, tc * TC:(tc + 1) * TC, :, :])
                        for a in nl.static_range(A):
                            for b in nl.static_range(B):
                                w_tile[a][b][...] = nisa.nc_transpose(w_raw[:, :, a, b])

                        # compute loop
                        for a in nl.static_range(A):
                            for b in nl.static_range(B):
                                # pack the halo JIT
                                local_pack = nl.copy(
                                    halo_buf[tc % 2][:, a:a + TH, b:b + TW]
                                ).reshape((TC, TH * TW))

                                # big iron
                                psum_tile[...] += nisa.nc_matmul(
                                    w_tile[a][b], 
                                    local_pack
                                )

                    # add bias
                    b_tile = nl.load(bias[tk * TK:(tk + 1) * TK])
                    psum_tile[...] = nl.add(psum_tile, b_tile)
                    
                    # write back result
                    nisa.dma_copy(
                        dst=out[n, tk * TK:(tk + 1) * TK, th * TH:(th + 1) * TH, tw * TW:(tw + 1) * TW], 
                        src=nl.copy(psum_tile).reshape((TK, TH, TW))
                    )

    return out