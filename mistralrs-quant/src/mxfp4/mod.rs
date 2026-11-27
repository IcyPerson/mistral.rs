use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, Result, Tensor};
use safetensors::tensor::Dtype;

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
use crate::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde,
    QuantizedSerdeType, Shard, ShardedVarBuilder, UqffReader, UqffTensor,
};

#[cfg(feature = "cuda")]
pub(crate) mod ffi;
#[cfg(feature = "metal")]
pub(crate) mod metal_ops;
#[cfg(feature = "cuda")]
pub(crate) mod ops;

/// MXFP4 block size (32 elements per scale)
pub const MXFP4_BLOCK_SIZE: usize = 32;

// Avoid Rayon scheduling overhead for tiny routed projections.
const CPU_MXFP4_PARALLEL_MIN_OUTPUTS: usize = 4096;

pub(crate) const N_BITS: usize = 4;

#[derive(Debug)]
pub struct MXFP4Layer {
    /// Packed FP4 weights: [N, K/2] or [num_experts, N, K/2]
    /// Each byte contains 2 FP4 values (low nibble = k, high nibble = k+1)
    #[allow(dead_code)]
    blocks: Tensor,
    /// E8M0 scales: [N, K/32] or [num_experts, N, K/32]
    /// Each byte is an 8-bit exponent with bias 127
    scales: Tensor,
    /// Optional bias: [N] or [num_experts, N]
    #[allow(dead_code)]
    bias: Option<Tensor>,
}

impl MXFP4Layer {
    pub(crate) fn inspect_uqff_header(layer: &UqffLayerHeaderView<'_>) -> Option<UqffHeaderMatch> {
        const WEIGHT_SUFFIXES: &[&str] = &["weight", "weight.format", "weight.scales"];
        if layer.exact_weight_suffixes(WEIGHT_SUFFIXES) && layer.scalar("weight.format", Dtype::U8)
        {
            Some(UqffHeaderMatch {
                serde_type: QuantizedSerdeType::Mxfp4,
            })
        } else {
            None
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        _tensors: &[UqffTensor],
        _prefix: &str,
    ) -> Result<String> {
        Ok("mxfp4".to_string())
    }
}

impl QuantMethod for MXFP4Layer {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Afq { .. } => unreachable!(),
            QuantMethodConfig::MXFP4 {
                blocks,
                scales,
                bias,
            } => Ok(Self {
                blocks,
                scales,
                bias,
            }),
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        self.dequantize_weights_to(self.blocks.device())
    }

    fn embedding_table(&self) -> Option<crate::EmbeddingTable<'_>> {
        use crate::embedding::{RowCodec, ScaleGrid, ScaleType};
        let (rows, k_half) = self.blocks.dims2().ok()?;
        Some(
            crate::EmbeddingTable::new(
                RowCodec::Fp4(ScaleGrid::per_row(ScaleType::E8m0Raw, MXFP4_BLOCK_SIZE)),
                rows,
                k_half * 2,
                &self.blocks,
            )
            .with_scales(&self.scales),
        )
    }

    #[allow(unused_variables)]
    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(x.device(), Device::Cuda(_)) && ffi::HAVE_MXFP4_GEMM_KERNELS {
            let orig_dims = x.dims().to_vec();
            let x_2d = if orig_dims.len() > 2 {
                let features = orig_dims[orig_dims.len() - 1];
                let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                x.reshape((batch_size, features))?
            } else {
                x.clone()
            };

            let result = ops::mxfp4_matmul(&x_2d, &self.blocks, &self.scales, self.bias.as_ref())?;

            if orig_dims.len() > 2 {
                let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                new_dims.push(result.dim(1)?);
                return result.reshape(new_dims);
            }
            return Ok(result);
        }

        #[cfg(feature = "metal")]
        {
            if x.device().is_metal() {
                let orig_dims = x.dims().to_vec();
                let x_2d = if orig_dims.len() > 2 {
                    let features = orig_dims[orig_dims.len() - 1];
                    let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                    x.reshape((batch_size, features))?
                } else {
                    x.clone()
                };

                let result =
                    metal_ops::mxfp4_matmul(&x_2d, &self.blocks, &self.scales, self.bias.as_ref())?;

                if orig_dims.len() > 2 {
                    let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                    new_dims.push(result.dim(1)?);
                    return result.reshape(new_dims);
                }
                return Ok(result);
            }
        }

        self.forward_dequantize(x)
    }

    fn routed_weights(&self) -> Option<crate::RoutedWeights> {
        let (experts, n, half_k) = self.blocks.dims3().ok()?;
        Some(crate::RoutedWeights::new(
            crate::RoutedCodec::Mxfp4,
            self.blocks.clone(),
            self.scales.clone(),
            None,
            self.bias.clone(),
            [experts, n, half_k * 2],
        ))
    }

    #[allow(unused_variables)]
    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        if let Some(routed) = self.routed_weights() {
            if let Some(output) = crate::try_routed_gather(x, indices, &routed)? {
                return Ok(output);
            }
        }
        #[cfg(feature = "cuda")]
        if matches!(x.device(), Device::Cuda(_)) && ffi::HAVE_MXFP4_GEMM_KERNELS {
            return ops::mxfp4_indexed_moe_gemm(
                x,
                &self.blocks,
                &self.scales,
                self.bias.as_ref(),
                indices,
            );
        }

        #[cfg(feature = "metal")]
        {
            if x.device().is_metal() {
                return metal_ops::mxfp4_indexed_moe_gemm(
                    x,
                    &self.blocks,
                    &self.scales,
                    self.bias.as_ref(),
                    indices,
                );
            }
        }

        self.gather_forward_dequantize(x, indices)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("MXFP4Layer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::BF16, self.scales.device().clone())
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        let mut shape = self.blocks.dims().to_vec();
        if let Some(last) = shape.last_mut() {
            *last = last.saturating_mul(2);
        }
        if shape.len() == 3 && request.ty.is_some_and(|ty| !ty.supports_stacked_gather()) {
            candle_core::bail!(
                "Cannot requantize packed MXFP4 expert weights to {}: that target does not support stacked expert gather.",
                request.ty.expect("rank-3 rejection requires an ISQ target")
            );
        }
        Ok(crate::plan_weight_isq(
            DType::BF16,
            self.scales.device().clone(),
            shape,
            request,
            true,
        ))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_none() || (dtype == Some(IsqType::MXFP4) && imatrix_weight.is_none()) {
            let blocks = self.blocks.to_device(&device)?;
            let scales = self.scales.to_device(&device)?;
            let bias = self
                .bias
                .as_ref()
                .map(|bias| bias.to_device(&device))
                .transpose()?;
            return Ok(Arc::new(Self::from_parts(blocks, scales, bias)));
        }

        let weight = self.dequantize_weights_to(&Device::Cpu)?;
        let bias = self
            .bias
            .as_ref()
            .map(|bias| bias.to_device(&Device::Cpu))
            .transpose()?;

        if weight.rank() == 3 {
            let Some(dtype) = dtype else {
                return Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
                    candle_nn::Linear::new(weight, bias),
                ))?)
                .apply_isq(None, device, n_quantized, imatrix_weight, guard);
            };

            if candle_core::quantized::GgmlDType::try_from(dtype).is_ok() {
                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let weight = crate::GgufMatMul::quantize_expert_stack(
                    &weight,
                    dtype,
                    imatrix_weight.as_deref(),
                    &device,
                    guard,
                )?;
                let bias = bias
                    .map(|bias| bias.to_dtype(DType::F32)?.to_device(&device))
                    .transpose()?;
                return Ok(Arc::new(crate::GgufMatMul::from_qtensor(weight, bias)));
            }

            if !dtype.supports_stacked_gather() {
                candle_core::bail!(
                    "Cannot requantize packed MXFP4 expert weights to {dtype}: that target does not support stacked expert gather."
                );
            }
        }

        Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
            candle_nn::Linear::new(weight, bias),
        ))?)
        .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }
}

impl MXFP4Layer {
    pub fn from_parts(blocks: Tensor, scales: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            blocks,
            scales,
            bias,
        }
    }

    fn from_uqff(reader: &UqffReader, key: &str, device: &Device, shard: Shard) -> Result<Self> {
        // Logical dims: blocks pack 2 FP4 input elements per byte along the last dim.
        let blocks_dims = reader.tensor_dims(&format!("{key}.weight"))?;
        let mut dims = blocks_dims.clone();
        *dims.last_mut().expect("MXFP4 blocks are non-empty") *= 2;
        let range = crate::uqff::shard_range(shard, &dims)?;
        let (blocks_range, scales_range) = match range {
            None => (None, None),
            Some((dim, start, len)) if dim == dims.len() - 1 => {
                if !start.is_multiple_of(MXFP4_BLOCK_SIZE) || !len.is_multiple_of(MXFP4_BLOCK_SIZE)
                {
                    candle_core::bail!(
                        "Sharding the MXFP4 packed dim requires alignment of {MXFP4_BLOCK_SIZE}: start {start}, len {len}."
                    );
                }
                (
                    Some((dim, start / 2, len / 2)),
                    Some((dim, start / MXFP4_BLOCK_SIZE, len / MXFP4_BLOCK_SIZE)),
                )
            }
            some => (some, some),
        };
        let blocks = reader.load_tensor_sharded(&format!("{key}.weight"), device, blocks_range)?;
        let scales =
            reader.load_tensor_sharded(&format!("{key}.weight.scales"), device, scales_range)?;
        let bias = reader.load_bias(key, device, range, dims.len())?;
        Ok(Self::from_parts(blocks, scales, bias))
    }

    /// Check if the device supports MXFP4 operations
    fn device_supported(_device: &Device) -> bool {
        #[cfg(feature = "cuda")]
        if matches!(_device, Device::Cuda(_)) {
            return ffi::HAVE_MXFP4_GEMM_KERNELS;
        }
        #[cfg(feature = "metal")]
        if _device.is_metal() {
            return true;
        }
        false
    }

    /// Quantize an unquantized weight tensor to MXFP4 format.
    /// weight shape: `[N, K]`, bias shape: `[N]` (optional)
    pub fn quantize(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight_f32 = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let dims = weight_f32.dims2()?;
        let (n, k) = (dims.0, dims.1);

        if k % MXFP4_BLOCK_SIZE != 0 {
            candle_core::bail!(
                "MXFP4 quantization requires K ({k}) divisible by block size ({MXFP4_BLOCK_SIZE})"
            );
        }

        let weight_data: Vec<f32> = weight_f32.flatten_all()?.to_vec1()?;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let k_half = k / 2;

        // Parallelize quantization across rows with rayon
        use rayon::prelude::*;
        let row_results: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
            .into_par_iter()
            .map(|row| {
                let row_offset = row * k;
                let mut row_packed = vec![0u8; k_half];
                let mut row_scales = vec![0u8; num_blocks_per_row];

                for (blk, row_scale) in row_scales.iter_mut().enumerate() {
                    let blk_start = row_offset + blk * MXFP4_BLOCK_SIZE;
                    let block = &weight_data[blk_start..blk_start + MXFP4_BLOCK_SIZE];

                    let max_abs = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

                    let scale = if max_abs == 0.0 {
                        127u8
                    } else {
                        let raw = (max_abs / 6.0).log2().floor() as i32 + 127;
                        raw.clamp(0, 254) as u8
                    };
                    *row_scale = scale;

                    let scale_factor = 2.0f32.powi(scale as i32 - 127);
                    let inv_scale = if scale_factor == 0.0 {
                        0.0
                    } else {
                        1.0 / scale_factor
                    };

                    for (elem, &val) in block.iter().enumerate() {
                        let nibble = Self::quantize_to_fp4(val * inv_scale);
                        let k_idx = blk * MXFP4_BLOCK_SIZE + elem;
                        let byte_idx = k_idx / 2;
                        if k_idx.is_multiple_of(2) {
                            row_packed[byte_idx] |= nibble;
                        } else {
                            row_packed[byte_idx] |= nibble << 4;
                        }
                    }
                }
                (row_packed, row_scales)
            })
            .collect();

        let mut packed = Vec::with_capacity(n * k_half);
        let mut scales = Vec::with_capacity(n * num_blocks_per_row);
        for (row_packed, row_scales) in row_results {
            packed.extend_from_slice(&row_packed);
            scales.extend_from_slice(&row_scales);
        }

        let blocks = Tensor::from_vec(packed, (n, k / 2), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let scales = Tensor::from_vec(scales, (n, num_blocks_per_row), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let bias = bias.map(|b| b.to_device(device)).transpose()?;

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    /// Quantize a single scaled value to the nearest FP4 E2M1 nibble (0..15).
    fn quantize_to_fp4(val: f32) -> u8 {
        // FP4 E2M1 positive values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        // Negative values are the same with sign bit set (indices 8..15)
        let sign = val < 0.0;
        let abs_val = val.abs();

        // Decision boundaries (midpoints between consecutive FP4 values)
        let nibble = if abs_val < 0.25 {
            0 // 0.0
        } else if abs_val < 0.75 {
            1 // 0.5
        } else if abs_val < 1.25 {
            2 // 1.0
        } else if abs_val < 1.75 {
            3 // 1.5
        } else if abs_val < 2.5 {
            4 // 2.0
        } else if abs_val < 3.5 {
            5 // 3.0
        } else if abs_val < 5.0 {
            6 // 4.0
        } else {
            7 // 6.0
        };

        if sign {
            nibble | 0x08
        } else {
            nibble
        }
    }

    pub fn linear_b(
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !Self::device_supported(vb.device()) {
            candle_core::bail!("MXFP4Layer requires CUDA or Metal device.");
        }

        let QuantizedConfig::MXFP4 {} = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let blocks = vb.get_with_hints_dtype(
            (out_dim, in_dim / 2),
            "blocks",
            Default::default(),
            DType::U8,
        )?;
        let scales = vb.get_with_hints_dtype(
            (out_dim, in_dim / MXFP4_BLOCK_SIZE),
            "scales",
            Default::default(),
            DType::U8,
        )?;

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    pub fn packed_linear_b(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !Self::device_supported(vb.device()) {
            candle_core::bail!("MXFP4Layer requires CUDA or Metal device.");
        }

        let QuantizedConfig::MXFP4 {} = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let blocks = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim / 2),
            "blocks",
            Default::default(),
            DType::U8,
        )?;
        let scales = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim / MXFP4_BLOCK_SIZE),
            "scales",
            Default::default(),
            DType::U8,
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    /// Load GPT-OSS style MXFP4 experts (combined gate_up_proj format).
    ///
    /// GPT-OSS stores tensors as:
    /// - `{name}_blocks`: [num_experts, out_dim, num_blocks, 16] where 16 bytes = 32 FP4 values
    /// - `{name}_scales`: [num_experts, out_dim, num_blocks]
    /// - `{name}_bias`: [num_experts, out_dim]
    ///
    /// This function loads and reshapes the 4D blocks tensor to 3D [num_experts, out_dim, in_dim/2].
    pub fn packed_gptoss_linear(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        name: &str,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let num_blocks = in_dim / MXFP4_BLOCK_SIZE;

        let blocks_4d = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, num_blocks, 16),
            &format!("{name}_blocks"),
            Default::default(),
            DType::U8,
        )?;

        let blocks = blocks_4d.reshape((num_local_experts, out_dim, num_blocks * 16))?;

        let scales = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, num_blocks),
            &format!("{name}_scales"),
            Default::default(),
            DType::U8,
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), &format!("{name}_bias"))?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    /// Combined FP4 × E8M0 dequant table: `DEQUANT_LUT[scale][nibble]`.
    /// For each of the 256 possible E8M0 scale values, stores the 16 possible
    /// dequantized values (FP4_LUT[nibble] * 2^(scale - 127)).
    /// This turns dequantization into a single table lookup per element.
    const DEQUANT_LUT: [[f32; 16]; 256] = {
        let mut lut = [[0.0f32; 16]; 256];
        let fp4: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut s = 0u32;
        while s < 256 {
            let scale_factor = f32::from_bits(s << 23);
            let mut n = 0;
            while n < 16 {
                lut[s as usize][n] = fp4[n] * scale_factor;
                n += 1;
            }
            s += 1;
        }
        lut
    };

    /// Dequantize MXFP4 weights to f32
    /// blocks: [num_experts, N, K/2] packed bytes
    /// scales: [num_experts, N, K/32] E8M0 scales
    /// Returns: [num_experts, N, K] f32 weights
    fn dequantize_weights_to(&self, device: &Device) -> Result<Tensor> {
        let blocks_dims = self.blocks.dims();

        let (num_experts, n, k_half) = if blocks_dims.len() == 3 {
            (blocks_dims[0], blocks_dims[1], blocks_dims[2])
        } else {
            (1, blocks_dims[0], blocks_dims[1])
        };
        let k = k_half * 2;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;

        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;

        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<u8> = scales_cpu.flatten_all()?.to_vec1()?;

        let mut weights = vec![0f32; num_experts * n * k];
        let half_block = MXFP4_BLOCK_SIZE / 2; // 16 packed bytes per block

        for expert in 0..num_experts {
            for row in 0..n {
                let blocks_row = expert * n * k_half + row * k_half;
                let scales_row = expert * n * num_blocks_per_row + row * num_blocks_per_row;
                let weights_row = expert * n * k + row * k;

                for blk in 0..num_blocks_per_row {
                    let scale = scales_data[scales_row + blk] as usize;
                    let dequant = &Self::DEQUANT_LUT[scale];
                    let blk_bytes = &blocks_data[blocks_row + blk * half_block..];
                    let w_out = &mut weights[weights_row + blk * MXFP4_BLOCK_SIZE..];

                    for byte_i in 0..half_block {
                        let packed = blk_bytes[byte_i];
                        w_out[byte_i * 2] = dequant[(packed & 0x0F) as usize];
                        w_out[byte_i * 2 + 1] = dequant[((packed >> 4) & 0x0F) as usize];
                    }
                }
            }
        }

        let shape = if blocks_dims.len() == 3 {
            vec![num_experts, n, k]
        } else {
            vec![n, k]
        };

        Tensor::from_vec(weights, shape.as_slice(), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(device)
    }

    /// CPU forward pass: blocked dequant + matmul to avoid full weight allocation.
    /// Processes MXFP4_BLOCK_SIZE (32) input columns at a time, dequantizing only
    /// the needed weight slice before accumulating partial results.
    fn forward_dequantize(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();

        let x_2d = if orig_dims.len() > 2 {
            let features = orig_dims[orig_dims.len() - 1];
            let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
            x.reshape((batch_size, features))?
        } else {
            x.clone()
        };

        let x_f32 = x_2d.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let (m, k) = x_f32.dims2()?;

        let blocks_dims = self.blocks.dims();
        let n = if blocks_dims.len() == 3 {
            blocks_dims[1]
        } else {
            blocks_dims[0]
        };
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let half_block = MXFP4_BLOCK_SIZE / 2;

        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;
        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<u8> = scales_cpu.flatten_all()?.to_vec1()?;
        let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;

        // output: [m, n], accumulate x @ W^T in blocks of 32 columns
        let mut output = vec![0f32; m * n];
        let k_half = k / 2;

        for blk in 0..num_blocks_per_row {
            let col_start = blk * MXFP4_BLOCK_SIZE;

            for row in 0..n {
                let scale = scales_data[row * num_blocks_per_row + blk] as usize;
                let dequant = &Self::DEQUANT_LUT[scale];
                let blk_bytes = &blocks_data[row * k_half + blk * half_block..];

                // Dequantize this block of 32 weights for this output row
                let mut w_block = [0f32; MXFP4_BLOCK_SIZE];
                for byte_i in 0..half_block {
                    let packed = blk_bytes[byte_i];
                    w_block[byte_i * 2] = dequant[(packed & 0x0F) as usize];
                    w_block[byte_i * 2 + 1] = dequant[((packed >> 4) & 0x0F) as usize];
                }

                // Accumulate dot product for all tokens against this weight block
                for token in 0..m {
                    let x_row = &x_data[token * k + col_start..];
                    let mut acc = 0f32;
                    for i in 0..MXFP4_BLOCK_SIZE {
                        acc += x_row[i] * w_block[i];
                    }
                    output[token * n + row] += acc;
                }
            }
        }

        let mut result = Tensor::from_vec(output, (m, n), &Device::Cpu)?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;

        if let Some(bias) = &self.bias {
            result = result.broadcast_add(bias)?;
        }

        if orig_dims.len() > 2 {
            let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
            new_dims.push(result.dim(1)?);
            result = result.reshape(new_dims)?;
        }

        Ok(result)
    }

    /// CPU MoE forward: blocked dequant per (token, expert) pair.
    /// Avoids dequantizing all experts, only touches the needed weight blocks.
    fn gather_forward_dequantize(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        use rayon::prelude::*;

        let x_dims = x.dims();
        let indices_dims = indices.dims();

        let (num_tokens, topk, k, x_has_topk) = if x_dims.len() == 2 {
            (x_dims[0], indices_dims[1], x_dims[1], false)
        } else {
            (x_dims[0], x_dims[1], x_dims[2], true)
        };

        let blocks_dims = self.blocks.dims();
        let num_experts = blocks_dims[0];
        let n = blocks_dims[1];
        let k_half = k / 2;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let half_block = MXFP4_BLOCK_SIZE / 2;

        let x_f32 = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;

        let indices_cpu = indices.to_device(&Device::Cpu)?.to_dtype(DType::U32)?;
        let indices_data: Vec<u32> = indices_cpu.flatten_all()?.to_vec1()?;

        // Materialize only experts selected by this routing step. The previous path copied
        // every expert's packed weights and scales into temporary Vecs on every decode token.
        // Narrow before moving to CPU so even a device fallback transfers only routed slices.
        let mut loaded = vec![false; num_experts];
        let mut expert_data: Vec<Option<(Vec<u8>, Vec<u8>)>> =
            std::iter::repeat_with(|| None).take(num_experts).collect();
        for &expert_id in &indices_data {
            let expert_idx = expert_id as usize;
            if expert_idx >= num_experts {
                candle_core::bail!(
                    "MXFP4 routed expert index {expert_idx} is out of range for {num_experts} experts"
                );
            }
            if !loaded[expert_idx] {
                let expert_blocks = self
                    .blocks
                    .narrow(0, expert_idx, 1)?
                    .to_device(&Device::Cpu)?;
                let expert_scales = self
                    .scales
                    .narrow(0, expert_idx, 1)?
                    .to_device(&Device::Cpu)?;
                expert_data[expert_idx] = Some((
                    expert_blocks.flatten_all()?.to_vec1()?,
                    expert_scales.flatten_all()?.to_vec1()?,
                ));
                loaded[expert_idx] = true;
            }
        }

        let bias_data: Option<Vec<f32>> = self
            .bias
            .as_ref()
            .map(|bias| {
                bias.to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1()
            })
            .transpose()?;
        let bias_is_per_expert = self.bias.as_ref().is_some_and(|bias| bias.rank() == 2);

        // Output is [num_tokens * topk, n]. Each element is independent, so decode
        // can use all available CPU workers even when there is only one input token.
        let mut output = vec![0f32; num_tokens * topk * n];
        let compute_output = |output_idx: usize, output_value: &mut f32| {
            let out_row = output_idx / n;
            let row = output_idx % n;
            let expert_idx = indices_data[out_row] as usize;
            let (blocks_data, scales_data) = expert_data[expert_idx]
                .as_ref()
                .expect("routed expert slices were loaded before compute");

            let x_offset = if x_has_topk {
                out_row * k
            } else {
                (out_row / topk) * k
            };

            let mut dot = 0f32;
            for blk in 0..num_blocks_per_row {
                let col_start = blk * MXFP4_BLOCK_SIZE;
                let x_blk =
                    &x_data[x_offset + col_start..x_offset + col_start + MXFP4_BLOCK_SIZE];
                let scale = scales_data[row * num_blocks_per_row + blk] as usize;
                let dequant = &Self::DEQUANT_LUT[scale];
                let block_start = row * k_half + blk * half_block;
                let blk_bytes = &blocks_data[block_start..block_start + half_block];

                let mut block_dot = 0f32;
                for byte_i in 0..half_block {
                    let packed = blk_bytes[byte_i];
                    let w0 = dequant[(packed & 0x0F) as usize];
                    let w1 = dequant[((packed >> 4) & 0x0F) as usize];
                    block_dot += x_blk[byte_i * 2] * w0 + x_blk[byte_i * 2 + 1] * w1;
                }
                // Preserve the reference path's per-block accumulation order.
                dot += block_dot;
            }

            if let Some(bias) = &bias_data {
                let bias_offset = if bias_is_per_expert { expert_idx * n } else { 0 };
                dot += bias[bias_offset + row];
            }
            *output_value = dot;
        };

        if output.len() >= CPU_MXFP4_PARALLEL_MIN_OUTPUTS {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, value)| compute_output(idx, value));
        } else {
            output
                .iter_mut()
                .enumerate()
                .for_each(|(idx, value)| compute_output(idx, value));
        }

        Tensor::from_vec(output, (num_tokens * topk, n), &Device::Cpu)?
            .to_device(x.device())?
            .to_dtype(x.dtype())?
            .reshape((num_tokens, topk, n))
    }
}

impl QuantizedSerde for MXFP4Layer {
    fn name(&self) -> &'static str {
        "mxfp4-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn uqff_type(&self) -> Option<IsqType> {
        Some(IsqType::MXFP4)
    }
    fn serialize_uqff(&self, prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        if ty != IsqType::MXFP4 {
            candle_core::bail!("Cannot serialize MXFP4 layer as {ty}; actual type is MXFP4.");
        }

        let mut data = vec![
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Mxfp4 as u8,
            ),
            UqffTensor::from_tensor(format!("{prefix}.weight"), &self.blocks)?,
            UqffTensor::from_tensor(format!("{prefix}.weight.scales"), &self.scales)?,
        ];
        if let Some(bias) = &self.bias {
            data.push(UqffTensor::from_tensor(format!("{prefix}.bias"), bias)?);
        }
        Ok(data)
    }
    fn deserialize_uqff(
        reader: &UqffReader,
        prefix: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self::from_uqff(reader, prefix, device, shard)?))
    }
    fn isq_type_from_uqff(_reader: &UqffReader, _prefix: &str) -> Result<IsqType> {
        Ok(IsqType::MXFP4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_HIDDEN_SIZE: usize = 64;
    const TEST_VOCAB_SIZE: usize = 7;

    fn test_layer() -> Result<Arc<dyn QuantMethod>> {
        let values = (0..TEST_VOCAB_SIZE * TEST_HIDDEN_SIZE)
            .map(|index| ((index % 29) as f32 - 14.0) / 3.0)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(values, (TEST_VOCAB_SIZE, TEST_HIDDEN_SIZE), &Device::Cpu)?;
        MXFP4Layer::quantize(&weight, None, &Device::Cpu)
    }

    fn expected_embedding(layer: &dyn QuantMethod, ids: &Tensor) -> Result<Tensor> {
        let weight = layer.dequantize_w()?;
        let mut shape = ids.dims().to_vec();
        shape.push(TEST_HIDDEN_SIZE);
        let ids = ids
            .to_device(weight.device())?
            .flatten_all()?
            .contiguous()?;
        weight.index_select(&ids, 0)?.reshape(shape)
    }

    fn stacked_test_layer() -> Result<Arc<MXFP4Layer>> {
        const EXPERTS: usize = 2;
        const OUTPUT: usize = 4;
        let blocks = Tensor::from_vec(
            vec![0x22u8; EXPERTS * OUTPUT * TEST_HIDDEN_SIZE / 2],
            (EXPERTS, OUTPUT, TEST_HIDDEN_SIZE / 2),
            &Device::Cpu,
        )?;
        let scales = Tensor::from_vec(
            vec![127u8; EXPERTS * OUTPUT * TEST_HIDDEN_SIZE / MXFP4_BLOCK_SIZE],
            (EXPERTS, OUTPUT, TEST_HIDDEN_SIZE / MXFP4_BLOCK_SIZE),
            &Device::Cpu,
        )?;
        let bias = Tensor::from_vec(
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            (EXPERTS, OUTPUT),
            &Device::Cpu,
        )?;
        Ok(Arc::new(MXFP4Layer::from_parts(blocks, scales, Some(bias))))
    }

    fn stacked_test_inputs() -> Result<(Tensor, Tensor)> {
        let input = Tensor::ones((2, 1, TEST_HIDDEN_SIZE), DType::F32, &Device::Cpu)?;
        let indices = Tensor::from_vec(vec![0u32, 1], (2, 1), &Device::Cpu)?;
        Ok((input, indices))
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn embedding_selects_quantized_rows() -> Result<()> {
        let layer = test_layer()?;
        let ids = Tensor::from_vec(vec![6u32, 1, 6, 3, 0, 4], (2, 3), &Device::Cpu)?;

        let output = layer.embedding_forward(&ids, DType::F32.into())?;
        let expected = expected_embedding(layer.as_ref(), &ids)?.to_dtype(DType::F32)?;

        assert_eq!(output.dims(), &[2, 3, TEST_HIDDEN_SIZE]);
        assert_eq!(output.dtype(), DType::F32);
        assert!(output.device().is_cpu());
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn uqff_loaded_embedding_selects_quantized_rows() -> Result<()> {
        let layer = test_layer()?;
        let mut tensors = crate::uqff_version_tensors();
        tensors.extend(layer.serialize_uqff("test.embedding", IsqType::MXFP4)?);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mistralrs-mxfp4-embedding-uqff-{}-{stamp}.uqff",
            std::process::id()
        ));
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .map_err(candle_core::Error::wrap)?;

        let reader = UqffReader::open(std::slice::from_ref(&path))?;
        let loaded = reader
            .load_linear("test.embedding", &Device::Cpu, Shard::default())?
            .unwrap();
        let ids = Tensor::from_vec(vec![5u32, 2, 1, 5], (2, 2), &Device::Cpu)?;
        let output = loaded.embedding_forward(&ids, DType::BF16.into())?;
        let expected = expected_embedding(layer.as_ref(), &ids)?;
        drop(reader);
        let _ = std::fs::remove_file(path);

        assert_eq!(output.dims(), &[2, 2, TEST_HIDDEN_SIZE]);
        assert_eq!(output.dtype(), DType::BF16);
        assert!(output.device().is_cpu());
        assert_eq!(
            output.flatten_all()?.to_vec1::<half::bf16>()?,
            expected.flatten_all()?.to_vec1::<half::bf16>()?
        );
        Ok(())
    }

    #[test]
    fn stacked_mxfp4_requantizes_to_ggml_with_bias() -> Result<()> {
        let layer = stacked_test_layer()?;
        let (input, indices) = stacked_test_inputs()?;
        let expected = layer.gather_forward(&input, &indices)?;
        let n_quantized = AtomicUsize::new(0);
        let converted = layer.apply_isq(
            Some(IsqType::Q4_0),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        let actual = converted.gather_forward(&input, &indices)?;

        assert_ne!(converted.name(), "mxfp4-layer");
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_close(&actual, &expected, 0.01)
    }

    #[test]
    fn stacked_mxfp4_preserves_packed_capture_and_exact_target() -> Result<()> {
        let source = stacked_test_layer()? as Arc<dyn QuantMethod>;
        let n_quantized = AtomicUsize::new(0);
        let captured = source.clone().apply_isq(
            None,
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        assert_eq!(captured.uqff_type(), Some(IsqType::MXFP4));
        assert!(captured.has_bias());

        let exact = source.apply_isq(
            Some(IsqType::MXFP4),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        assert_eq!(exact.uqff_type(), Some(IsqType::MXFP4));
        assert!(exact.has_bias());
        Ok(())
    }

    #[test]
    fn stacked_mxfp4_requantizes_to_afq_with_selected_bias() -> Result<()> {
        let layer = stacked_test_layer()?;
        let (input, indices) = stacked_test_inputs()?;
        let expected = layer.gather_forward(&input, &indices)?;
        let n_quantized = AtomicUsize::new(0);
        let converted = layer.apply_isq(
            Some(IsqType::AFQ4),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        let actual = converted.gather_forward(&input, &indices)?;

        assert_ne!(converted.name(), "mxfp4-layer");
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_close(&actual, &expected, 0.01)
    }

    #[test]
    fn stacked_mxfp4_requantizes_to_hqq_and_fp8() -> Result<()> {
        requantize_stacked_experts(Device::Cpu)?;
        #[cfg(feature = "cuda")]
        requantize_stacked_experts(Device::new_cuda_with_stream(0)?)?;
        Ok(())
    }

    fn requantize_stacked_experts(device: Device) -> Result<()> {
        let (input, indices) = stacked_test_inputs()?;
        let input = input.to_device(&device)?;
        let indices = indices.to_device(&device)?;
        for ty in [IsqType::HQQ4, IsqType::HQQ8, IsqType::F8E4M3] {
            let layer = stacked_test_layer()?;
            let expected = layer
                .gather_forward(
                    &input.to_device(&Device::Cpu)?,
                    &indices.to_device(&Device::Cpu)?,
                )?
                .to_device(&device)?;
            let request = crate::IsqRequest {
                ty: Some(ty),
                device: device.clone(),
                has_imatrix: false,
                capture: crate::IsqCaptureMode::Immediate,
                consumer: crate::IsqConsumer::RuntimeSwap,
                module_key: "experts".to_string(),
            };
            layer.plan_isq(&request)?;
            let n_quantized = AtomicUsize::new(0);
            let converted = layer.apply_isq(
                Some(ty),
                device.clone(),
                &n_quantized,
                None,
                QuantizeOntoGuard::new(),
            )?;
            assert_ne!(converted.name(), "mxfp4-layer");
            assert!(converted.has_bias());
            assert_close(
                &converted.gather_forward(&input, &indices)?,
                &expected,
                0.05,
            )?;
        }
        Ok(())
    }

    #[test]
    fn stacked_mxfp4_rejects_targets_without_expert_gather() -> Result<()> {
        let layer = stacked_test_layer()?;
        let request = crate::IsqRequest {
            ty: Some(IsqType::F8Q8),
            device: Device::Cpu,
            has_imatrix: false,
            capture: crate::IsqCaptureMode::Immediate,
            consumer: crate::IsqConsumer::UqffWrite,
            module_key: "experts".to_string(),
        };
        let plan_error = layer.plan_isq(&request).unwrap_err();
        assert!(plan_error
            .to_string()
            .contains("does not support stacked expert gather"));
        let error = layer
            .apply_isq(
                Some(IsqType::F8Q8),
                Device::Cpu,
                &AtomicUsize::new(0),
                None,
                QuantizeOntoGuard::new(),
            )
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("packed MXFP4 expert weights"));
        assert!(message.contains("does not support stacked expert gather"));
        Ok(())
    }

    fn serial_gather_forward_reference(
        layer: &MXFP4Layer,
        x: &Tensor,
        indices: &Tensor,
    ) -> Result<Tensor> {
        let x_dims = x.dims();
        let indices_dims = indices.dims();
        let (num_tokens, topk, k, x_has_topk) = if x_dims.len() == 2 {
            (x_dims[0], indices_dims[1], x_dims[1], false)
        } else {
            (x_dims[0], x_dims[1], x_dims[2], true)
        };
        let blocks_dims = layer.blocks.dims();
        let num_experts = blocks_dims[0];
        let n = blocks_dims[1];
        let k_half = k / 2;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let half_block = MXFP4_BLOCK_SIZE / 2;

        let blocks_cpu = layer.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = layer.scales.to_device(&Device::Cpu)?;
        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<u8> = scales_cpu.flatten_all()?.to_vec1()?;
        let x_f32 = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;
        let indices_cpu = indices.to_device(&Device::Cpu)?.to_dtype(DType::U32)?;
        let indices_data: Vec<u32> = indices_cpu.flatten_all()?.to_vec1()?;
        let bias_data: Option<Vec<f32>> = layer
            .bias
            .as_ref()
            .map(|bias| {
                bias.to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1()
            })
            .transpose()?;
        let bias_is_per_expert = layer.bias.as_ref().is_some_and(|bias| bias.rank() == 2);

        let mut output = vec![0f32; num_tokens * topk * n];
        for token_idx in 0..num_tokens {
            for slot_idx in 0..topk {
                let out_row = token_idx * topk + slot_idx;
                let expert_idx = indices_data[out_row] as usize;
                assert!(expert_idx < num_experts);
                let x_offset = if x_has_topk {
                    out_row * k
                } else {
                    token_idx * k
                };
                let expert_blocks_base = expert_idx * n * k_half;
                let expert_scales_base = expert_idx * n * num_blocks_per_row;

                for blk in 0..num_blocks_per_row {
                    let col_start = blk * MXFP4_BLOCK_SIZE;
                    let x_blk =
                        &x_data[x_offset + col_start..x_offset + col_start + MXFP4_BLOCK_SIZE];

                    for row in 0..n {
                        let scale =
                            scales_data[expert_scales_base + row * num_blocks_per_row + blk]
                                as usize;
                        let dequant = &MXFP4Layer::DEQUANT_LUT[scale];
                        let block_start =
                            expert_blocks_base + row * k_half + blk * half_block;
                        let blk_bytes = &blocks_data[block_start..block_start + half_block];
                        let mut dot = 0f32;
                        for byte_i in 0..half_block {
                            let packed = blk_bytes[byte_i];
                            dot += x_blk[byte_i * 2] * dequant[(packed & 0x0F) as usize]
                                + x_blk[byte_i * 2 + 1]
                                    * dequant[((packed >> 4) & 0x0F) as usize];
                        }
                        output[out_row * n + row] += dot;
                    }
                }

                if let Some(bias) = &bias_data {
                    let bias_offset = if bias_is_per_expert { expert_idx * n } else { 0 };
                    for row in 0..n {
                        output[out_row * n + row] += bias[bias_offset + row];
                    }
                }
            }
        }

        Tensor::from_vec(output, (num_tokens * topk, n), &Device::Cpu)?
            .to_device(x.device())?
            .to_dtype(x.dtype())?
            .reshape((num_tokens, topk, n))
    }

    fn gather_test_layer(
        experts: usize,
        n: usize,
        k: usize,
        per_expert_bias: bool,
    ) -> Result<MXFP4Layer> {
        let blocks = Tensor::from_vec(
            (0..experts * n * (k / 2))
                .map(|index| ((index * 37 + 11) % 256) as u8)
                .collect::<Vec<_>>(),
            (experts, n, k / 2),
            &Device::Cpu,
        )?;
        let scales = Tensor::from_vec(
            vec![127u8; experts * n * (k / MXFP4_BLOCK_SIZE)],
            (experts, n, k / MXFP4_BLOCK_SIZE),
            &Device::Cpu,
        )?;
        let bias = if per_expert_bias {
            Some(Tensor::from_vec(
                (0..experts * n)
                    .map(|index| (index % 31) as f32 * 0.01)
                    .collect::<Vec<_>>(),
                (experts, n),
                &Device::Cpu,
            )?)
        } else {
            Some(Tensor::from_vec(
                (0..n)
                    .map(|index| (index % 17) as f32 * 0.01)
                    .collect::<Vec<_>>(),
                (n,),
                &Device::Cpu,
            )?)
        };
        Ok(MXFP4Layer::from_parts(blocks, scales, bias))
    }

    #[test]
    fn cpu_mxfp4_gather_matches_serial_reference() -> Result<()> {
        const K: usize = 64;
        const N: usize = 2048;
        let indices = Tensor::new(&[[1u32, 3u32]], &Device::Cpu)?;
        let input = Tensor::from_vec(
            (0..K).map(|index| (index as f32 % 13.0) / 7.0).collect::<Vec<_>>(),
            (1, K),
            &Device::Cpu,
        )?;
        let input_with_topk = Tensor::from_vec(
            (0..2 * K)
                .map(|index| (index as f32 % 19.0) / 11.0)
                .collect::<Vec<_>>(),
            (1, 2, K),
            &Device::Cpu,
        )?;

        // Output size is 4096, exercising the Rayon path for a single-token decode.
        for per_expert_bias in [false, true] {
            let layer = gather_test_layer(4, N, K, per_expert_bias)?;
            let actual = layer.gather_forward(&input, &indices)?;
            let expected = serial_gather_forward_reference(&layer, &input, &indices)?;
            assert_close(&actual, &expected, 1e-5)?;

            let actual_topk = layer.gather_forward(&input_with_topk, &indices)?;
            let expected_topk =
                serial_gather_forward_reference(&layer, &input_with_topk, &indices)?;
            assert_close(&actual_topk, &expected_topk, 1e-5)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "manual CPU microbenchmark; run with --ignored --nocapture"]
    fn cpu_mxfp4_gather_benchmark() -> Result<()> {
        const EXPERTS: usize = 32;
        const N: usize = 2048;
        const K: usize = 2048;
        const TOPK: usize = 4;
        const ITERATIONS: usize = 3;

        let layer = MXFP4Layer::from_parts(
            Tensor::from_vec(
                vec![0x22u8; EXPERTS * N * (K / 2)],
                (EXPERTS, N, K / 2),
                &Device::Cpu,
            )?,
            Tensor::from_vec(
                vec![127u8; EXPERTS * N * (K / MXFP4_BLOCK_SIZE)],
                (EXPERTS, N, K / MXFP4_BLOCK_SIZE),
                &Device::Cpu,
            )?,
            None,
        );
        let input = Tensor::ones((1, K), DType::F32, &Device::Cpu)?;
        let indices = Tensor::new(&[[0u32, 7u32, 15u32, 31u32]], &Device::Cpu)?;

        let reference = serial_gather_forward_reference(&layer, &input, &indices)?;
        let optimized = layer.gather_forward(&input, &indices)?;
        assert_close(&optimized, &reference, 1e-5)?;

        let mut serial_times = Vec::with_capacity(ITERATIONS);
        let mut optimized_times = Vec::with_capacity(ITERATIONS);
        for _ in 0..ITERATIONS {
            let start = std::time::Instant::now();
            let _ = serial_gather_forward_reference(&layer, &input, &indices)?;
            serial_times.push(start.elapsed().as_secs_f64() * 1000.0);

            let start = std::time::Instant::now();
            let _ = layer.gather_forward(&input, &indices)?;
            optimized_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        serial_times.sort_by(f64::total_cmp);
        optimized_times.sort_by(f64::total_cmp);

        let serial_ms = serial_times[ITERATIONS / 2];
        let optimized_ms = optimized_times[ITERATIONS / 2];
        let bytes_per_expert = N * (K / 2 + K / MXFP4_BLOCK_SIZE);
        let full_copy_mib = (EXPERTS * bytes_per_expert) as f64 / (1024.0 * 1024.0);
        let routed_copy_mib = (TOPK * bytes_per_expert) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "MXFP4 CPU gather: serial_median_ms={serial_ms:.2}, optimized_median_ms={optimized_ms:.2}, speedup={:.2}x, copied_per_call={full_copy_mib:.1}MiB->{routed_copy_mib:.1}MiB",
            serial_ms / optimized_ms,
        );
        Ok(())
    }
}
