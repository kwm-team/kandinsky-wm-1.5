# Copyright 2024 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is modified based on autoencoder_kl_ltx.py. We thank the original authors for their work.

import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import convert_conv3d_weight_memory_format as convert_conv3d_weight

from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin

# Benchmark controls compilation explicitly; upstream decorator removed.
def magi_compile(*args, **kwargs):
    return lambda fn: fn


USE_TILING = os.environ.get("VAE_TILING", "0") == "1"
if USE_TILING:
    print("use vae tiling")


class SimpleConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(*args, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)



class TurboVAEDCausalConv3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]] = 3,
            stride: Union[int, Tuple[int, int, int]] = 1,
            dilation: Union[int, Tuple[int, int, int]] = 1,
            groups: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.temporal_padding = self.kernel_size[0] // 2
        spatial_padding = (self.kernel_size[1] // 2, self.kernel_size[2] // 2)

        self.conv = nn.Conv3d(
            in_channels, out_channels, self.kernel_size, stride, padding=(0, *spatial_padding),
            dilation=dilation, groups=groups
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.temporal_padding > 0:
            pad_left = hidden_states[:, :, :1].repeat(1, 1, self.temporal_padding, 1, 1)
            pad_right = hidden_states[:, :, -1:].repeat(1, 1, self.temporal_padding, 1, 1)
            hidden_states = torch.cat((pad_left, hidden_states, pad_right), dim=2)

        return self.conv(hidden_states)


class TurboVAEDCausalDepthwiseSeperableConv3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]] = 3,
            stride: Union[int, Tuple[int, int, int]] = 1,
            dilation: Union[int, Tuple[int, int, int]] = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)

        self.temporal_padding = self.kernel_size[0] // 2
        spatial_padding = (self.kernel_size[1] // 2, self.kernel_size[2] // 2)

        # Depthwise Convolution
        self.depthwise_conv = nn.Conv3d(
            in_channels, in_channels, self.kernel_size, self.stride, padding=(0, *spatial_padding),
            dilation=self.dilation, groups=in_channels
        )

        # Pointwise Convolution
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        if self.temporal_padding > 0:
            pad_left = hidden_states[:, :, :1].repeat(1, 1, self.temporal_padding, 1, 1)
            pad_right = hidden_states[:, :, -1:].repeat(1, 1, self.temporal_padding, 1, 1)
            hidden_states = torch.cat((pad_left, hidden_states, pad_right), dim=2)

        # Apply depthwise convolution
        hidden_states = self.depthwise_conv(hidden_states)
        # Apply pointwise convolution
        hidden_states = self.pointwise_conv(hidden_states)

        return hidden_states


class MyRMSNorm(nn.Module):

    def __init__(self, axis: int, eps: float):
        super().__init__()
        self.eps = eps
        self.axis = axis

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(self.axis, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(input_dtype)

        return hidden_states


class TurboVAEDResnetBlock3d(nn.Module):
    r"""
    A 3D ResNet block used in the TurboVAED model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            eps: float = 1e-6,
            is_upsampler_modified: bool = False,
            is_dw_conv: bool = False,
            dw_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        self.is_upsampler_modified = is_upsampler_modified
        self.conv_operation = TurboVAEDCausalConv3d if not is_dw_conv else TurboVAEDCausalDepthwiseSeperableConv3d
        self.kernel_size = 3 if not is_dw_conv else dw_kernel_size

        self.norm = MyRMSNorm(axis=1, eps=1e-8)
        self.conv1 = self.conv_operation(in_channels, out_channels, self.kernel_size)
        self.conv2 = self.conv_operation(out_channels, out_channels, self.kernel_size)

        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            self.conv_shortcut = SimpleConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs

        hidden_states = self.norm(hidden_states)
        if self.is_upsampler_modified:
            hidden_states = F.relu(hidden_states, inplace=True)
        else:
            hidden_states = F.silu(hidden_states, inplace=True)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = F.silu(hidden_states, inplace=True)
        hidden_states = self.conv2(hidden_states)

        if self.norm3 is not None:
            inputs = self.norm3(inputs.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class TurboVAEDUpsampler3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            stride: Union[int, Tuple[int, int, int]] = 1,
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        up_t, up_h, up_w = self.stride
        out_channels = in_channels * up_t * up_h * up_w

        self.conv = TurboVAEDCausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_frames, height, width = hidden_states.shape
        up_t, up_s = self.stride[:2]

        hidden_states = (
            self.conv(hidden_states)
            .reshape(batch_size, -1, up_s, up_s, up_t, num_frames, height, width)
            .permute(0, 1, 5, 4, 6, 2, 7, 3)
            .reshape(batch_size, -1, num_frames * up_t, height * up_s, width * up_s)[:, :, up_t - 1:]
        )

        return hidden_states


# Adapted from diffusers.models.autoencoders.autoencoder_kl_cogvideox.CogVideoMidBlock3d
class TurboVAEDMidBlock3d(nn.Module):
    r"""
    A middle block used in the TurboVAED model.

    Args:
        in_channels (`int`):
            Number of input channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self, in_channels: int, num_layers: int = 1, resnet_eps: float = 1e-6,
        is_dw_conv: bool = False, dw_kernel_size: int = 3,
    ) -> None:

        super().__init__()
        self.resnets = nn.Sequential(*[
            TurboVAEDResnetBlock3d(
                in_channels=in_channels, out_channels=in_channels, eps=resnet_eps,
                is_dw_conv=is_dw_conv, dw_kernel_size=dw_kernel_size
            ) for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Forward method of the `LTXMidBlock3D` class."""
        return self.resnets(hidden_states)


class TurboVAEDUpBlock3d(nn.Module):
    r"""
    Up block used in the TurboVAED model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        spatio_temporal_scale (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
            Whether or not to downsample across temporal dimension.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            spatio_temporal_scale: bool = True,
            is_dw_conv: bool = False,
            dw_kernel_size: int = 3,
            spatio_only: bool = False,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = TurboVAEDResnetBlock3d(
                in_channels=in_channels, out_channels=out_channels, eps=resnet_eps,
                is_dw_conv=is_dw_conv, dw_kernel_size=dw_kernel_size
            )

        self.upsamplers = None
        if spatio_temporal_scale:
            stride_up = (2, 2, 2) if not spatio_only else (1, 2, 2)
            self.upsamplers = nn.ModuleList([
                TurboVAEDUpsampler3d(out_channels, stride_up)
            ])

        self.resnets = nn.Sequential(*[
            TurboVAEDResnetBlock3d(
                in_channels=out_channels, out_channels=out_channels, eps=resnet_eps,
                is_dw_conv=is_dw_conv, dw_kernel_size=dw_kernel_size,
                is_upsampler_modified=(spatio_temporal_scale),
            ) for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return self.resnets(hidden_states)

class TurboVAEDDecoder3d(nn.Module):
    r"""
    The `TurboVAEDDecoder3d` layer of a variational autoencoder that decodes its latent representation into an output
    sample.

    Args:
        in_channels (`int`, defaults to 128):
            Number of latent channels.
        out_channels (`int`, defaults to 3):
            Number of output channels.
        block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512)`):
            The number of output channels for each block.
        spatio_temporal_scaling (`Tuple[bool, ...], defaults to `(True, True, True, False)`:
            Whether a block should contain spatio-temporal upscaling layers or not.
        layers_per_block (`Tuple[int, ...]`, defaults to `(4, 3, 3, 3, 4)`):
            The number of layers per block.
        resnet_norm_eps (`float`, defaults to `1e-6`):
            Epsilon value for ResNet normalization layers.
    """

    def __init__(
            self,
            in_channels: int = 16,
            out_channels: int = 3,
            block_out_channels: Tuple[int, ...] = (16, 32, 256, 512),
            spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
            layers_per_block: Tuple[int, ...] = (5, 4, 4, 4, 5),
            resnet_norm_eps: float = 1e-6,
            decoder_is_dw_conv: Tuple[bool, ...] = (False, False, False, True, True),
            decoder_dw_kernel_size: int = 5,
            spatio_only: Tuple[bool, ...] = (True, False, False, False),
    ) -> None:
        super().__init__()

        self.out_channels = out_channels

        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        decoder_is_dw_conv = tuple(reversed(decoder_is_dw_conv))
        spatio_only = tuple(reversed(spatio_only))
        output_channel = block_out_channels[0]

        self.conv_in = TurboVAEDCausalConv3d(
            in_channels=in_channels, out_channels=output_channel, kernel_size=3, stride=1
        )

        self.mid_block = TurboVAEDMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            is_dw_conv=decoder_is_dw_conv[0],
            dw_kernel_size=decoder_dw_kernel_size,
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        num_block_out_channels = len(block_out_channels)
        for i in range(num_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            up_block = TurboVAEDUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                is_dw_conv=decoder_is_dw_conv[i + 1],
                dw_kernel_size=decoder_dw_kernel_size,
                spatio_only=spatio_only[i],
            )

            self.up_blocks.append(up_block)

        self.norm_out = MyRMSNorm(axis=1, eps=1e-8)
        self.conv_out = TurboVAEDCausalConv3d(output_channel, self.out_channels,
                                              kernel_size=3, stride=1)

    @magi_compile(model_tag="turbo_vae_decoder", dynamic_arg_dims={"hidden_states": [3, 4]})
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = hidden_states.to(memory_format=torch.contiguous_format)
        hidden_states = self.conv_in(hidden_states)

        hidden_states = self.mid_block(hidden_states)
        hidden_states = self.up_blocks[0](hidden_states)

        hidden_states = hidden_states.to(memory_format=torch.channels_last_3d)
        for up_block in self.up_blocks[1:]:
            hidden_states = up_block(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states, inplace=True)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class AutoencoderKLTurboVAED(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images. Used in
    [LTX](https://huggingface.co/Lightricks/LTX-Video).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Args:
        in_channels (`int`, defaults to `3`):
            Number of input channels.
        out_channels (`int`, defaults to `3`):
            Number of output channels.
        latent_channels (`int`, defaults to `128`):
            Number of latent channels.
        block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512)`):
            The number of output channels for each block.
        spatio_temporal_scaling (`Tuple[bool, ...], defaults to `(True, True, True, False)`:
            Whether a block should contain spatio-temporal downscaling or not.
        layers_per_block (`Tuple[int, ...]`, defaults to `(4, 3, 3, 3, 4)`):
            The number of layers per block.
        resnet_norm_eps (`float`, defaults to `1e-6`):
            Epsilon value for ResNet normalization layers.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            out_channels: int = 3,
            latent_channels: int = 16,
            decoder_block_out_channels: Tuple[int, ...] = (16, 32, 256, 512),
            decoder_layers_per_block: Tuple[int, ...] = (5, 4, 4, 4, 5),
            spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
            decoder_spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
            resnet_norm_eps: float = 1e-6,
            spatial_compression_ratio: Optional[int] = 8,
            temporal_compression_ratio: Optional[int] = 4,
            decoder_is_dw_conv: Tuple[bool, ...] = (False, False, False, True, True),
            decoder_dw_kernel_size: int = 5,
            decoder_spatio_only: Tuple[bool, ...] = (True, False, False, False),
    ) -> None:
        super().__init__()

        self.decoder = TurboVAEDDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decoder_block_out_channels,
            spatio_temporal_scaling=decoder_spatio_temporal_scaling,
            layers_per_block=decoder_layers_per_block,
            resnet_norm_eps=resnet_norm_eps,
            decoder_is_dw_conv=decoder_is_dw_conv,
            decoder_dw_kernel_size=decoder_dw_kernel_size,
            spatio_only=decoder_spatio_only,
        )

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = USE_TILING

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 512
        self.tile_sample_min_width  = 512

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width  = 448

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        _, _, _, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(self, z: torch.Tensor, return_dict: bool = True, blend_size: int = 32) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """

        if self.use_tiling:
            tile_size, tile_stride = self.get_dec_optimal_tiling(z.shape, blend_size)
            self.apply_tiling(tile_size, tile_stride)

        decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned,
                otherwise a plain `tuple` is returned.
        """

        N, _, frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width  = width  * self.spatial_compression_ratio
        sample_frames = (frames - 1) * self.temporal_compression_ratio + 1
        context = {"dtype": z.dtype, "device": z.device}

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width  = self.tile_sample_min_width  - self.tile_sample_stride_width
        blend_latent_height = blend_height // self.spatial_compression_ratio
        blend_latent_width  = blend_width  // self.spatial_compression_ratio

        output = torch.empty(
            (N, self.out_channels, sample_frames, sample_height, sample_width),
            **context, memory_format=torch.channels_last_3d
        )
        output.fill_(0)

        range_right_height = torch.arange(blend_height, **context).view(1, 1, 1, -1, 1) / blend_height
        range_right_width  = torch.arange(blend_width,  **context).view(1, 1, 1, 1, -1) / blend_width
        range_left_height, range_left_width = (1 - range_right_height), (1 - range_right_width)

        height_indices = list(range(0, height - blend_latent_height, self.tile_latent_stride_height))
        width_indices  = list(range(0, width  - blend_latent_width,  self.tile_latent_stride_width))

        i_begin = 0
        for i in height_indices:
            j_begin = 0
            for j in width_indices:
                tile = z[:, :, :, i:i + self.tile_latent_min_height, j:j + self.tile_latent_min_width]
                # tile = self.post_quant_conv(tile)
                tile = self.decoder(tile)

                if i > height_indices[0]:
                    tile[:, :, :, :blend_height, :]  *= range_right_height
                if i < height_indices[-1]:
                    tile[:, :, :, -blend_height:, :] *= range_left_height

                if j > width_indices[0]:
                    tile[:, :, :, :, :blend_width]  *= range_right_width
                if j < width_indices[-1]:
                    tile[:, :, :, :, -blend_width:] *= range_left_width

                tile_height, tile_width = tile.shape[3:]
                i_end, j_end = (i_begin + tile_height), (j_begin + tile_width)
                output[:, :, :, i_begin:i_end, j_begin:j_end] += tile
                j_begin = j_end - blend_width
            i_begin = i_end - blend_height

        if not return_dict:
            return (output,)
        return DecoderOutput(sample=output)

    def get_dec_optimal_tiling(
        self, shape: Union[torch.Size, Tuple[int, int]], blend_size: int = 32
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Returns optimal tiling for given shape."""
        height, width = shape[-2:]
        height, width = 8 * height, 8 * width

        hs, ws = (height - blend_size) // 2, (width - blend_size) // 2
        ht, wt = hs + blend_size, ws + blend_size
        # print(f"Dec tiling", (ht, wt), (hs, ws))
        return (ht, wt), (hs, ws)

    def apply_tiling(self, tile: Tuple[int, int], stride: Tuple[int, int]):
        """Applies tiling."""
        ht, wt = tile
        hs, ws = stride

        self.use_tiling = USE_TILING
        self.tile_sample_min_height = ht
        self.tile_sample_min_width = wt
        self.tile_sample_stride_height = hs
        self.tile_sample_stride_width = ws
        self.tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        self.tile_latent_min_width  = self.tile_sample_min_width  // self.spatial_compression_ratio
        self.tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        self.tile_latent_stride_width  = self.tile_sample_stride_width  // self.spatial_compression_ratio

    def convert_conv3d_weights(self):

        self.decoder = convert_conv3d_weight(self.decoder, torch.contiguous_format)
        for i, up_block in enumerate(self.decoder.up_blocks[1:], start=1):
            self.decoder.up_blocks[i] = convert_conv3d_weight(up_block, torch.channels_last_3d)
        self.decoder.conv_out = convert_conv3d_weight(self.decoder.conv_out, torch.channels_last_3d)
