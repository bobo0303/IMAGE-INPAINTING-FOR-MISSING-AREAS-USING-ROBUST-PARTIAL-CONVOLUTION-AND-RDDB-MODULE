import numpy
import chainer.functions as F
import chainer
import cupy


def update_mask_small(mask, channel):
    B, C, H, W = mask.shape

    one_channel_kernel_size = cupy.zeros((3, 3))
    one_channel_kernel_size[0, 1] = 1
    one_channel_kernel_size[1, 0] = 1
    one_channel_kernel_size[1, 1] = 1
    one_channel_kernel_size[1, 2] = 1
    one_channel_kernel_size[2, 1] = 1

    kernel_size = cupy.zeros([B, C, 3, 3])
    for i in range(0, B):
        kernel_size[i] = one_channel_kernel_size
    output_mask = F.convolution_2d(mask, kernel_size, pad=1)
    output_mask = cupy.sign(abs(output_mask.data))
    final_mask = cupy.zeros((mask.shape[0], channel, mask.shape[2], mask.shape[3]), dtype=cupy.float32)
    for i in range(0, mask.shape[0]):
        final_mask[i] = output_mask[i, 0]
    return final_mask


def update_mask_big(mask, channel):
    final_mask = cupy.zeros((mask.shape[0], channel, mask.shape[2], mask.shape[3]), dtype=cupy.float32)
    output_mask = cupy.sign(abs(mask.data) + 1)
    for i in range(0, mask.shape[0]):
        final_mask[i] = output_mask[i, 0]
    return final_mask
