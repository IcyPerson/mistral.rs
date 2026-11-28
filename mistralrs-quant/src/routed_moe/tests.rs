use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Linear;
use float8::F8E4M3;

use super::*;
use crate::{mxfp8::Mxfp8Linear, QuantMethodConfig, UnquantLinear};

const EXPERTS: usize = 2;
const FEATURES: usize = 32;

fn projection(values: [f32; EXPERTS], quantized: bool) -> Result<Arc<dyn QuantMethod>> {
    let device = &Device::Cpu;
    let weight = Tensor::from_vec(
        values
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, FEATURES * FEATURES))
            .collect::<Vec<_>>(),
        (EXPERTS, FEATURES, FEATURES),
        device,
    )?;
    if quantized {
        let weight = Tensor::from_vec(
            weight
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(F8E4M3::from_f32)
                .collect::<Vec<_>>(),
            weight.shape(),
            device,
        )?;
        Ok(Arc::new(Mxfp8Linear::from_parts(
            weight,
            Tensor::full(127u8, (EXPERTS, FEATURES, 1), device)?,
            None,
            DType::F32,
        )?))
    } else {
        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(weight, None)),
        )?))
    }
}

#[test]
fn cpu_mxfp8_and_mixed_projections_use_canonical_routes() -> Result<()> {
    let device = &Device::Cpu;
    let mut values = vec![0f32; 2 * FEATURES];
    values[0] = 1.0;
    values[FEATURES] = 2.0;
    let xs = Tensor::from_vec(values, (2, FEATURES), device)?;
    let indices = Tensor::new(&[[0u32, 1], [1, 0]], device)?;
    let routing = Tensor::new(&[[0.25f32, 0.75], [0.75, 0.25]], device)?;
    for mask in [
        [true; 3],
        [true, false, false],
        [false, true, false],
        [false, false, true],
    ] {
        let gate = projection([1.0, 2.0], mask[0])?;
        let up = projection([3.0, 4.0], mask[1])?;
        let down = projection([1.0, 2.0], mask[2])?;
        let experts = RoutedExperts {
            gate: &*gate,
            up: &*up,
            down: &*down,
            num_experts: EXPERTS,
        };
        let result = experts.forward(
            RoutedInput {
                xs: &xs,
                indices: &indices,
                routing: &routing,
                output_dtype: DType::F32,
                lora: None,
            },
            RoutedActivation {
                fused: Some(GluActivationType::Relu),
                fallback: &candle_nn::Activation::Relu,
            },
        )?;
        assert_eq!(
            result.to_vec2::<f32>()?,
            vec![vec![408.0; FEATURES], vec![1632.0; FEATURES]]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_gguf_biases_are_preserved_in_decode_and_prefill() -> Result<()> {
    use crate::GgufMatMul;
    use candle_core::quantized::{GgmlDType, QTensor};

    const HIDDEN: usize = 64;
    let device = Device::new_cuda(0)?;
    let projection = |biases: [f32; EXPERTS]| -> Result<Arc<dyn QuantMethod>> {
        let weights = Tensor::ones((EXPERTS, HIDDEN, HIDDEN), DType::F32, &Device::Cpu)?;
        let weights = QTensor::quantize_onto(&weights, GgmlDType::Q4_0, &device)?;
        let biases = biases
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, HIDDEN))
            .collect::<Vec<_>>();
        Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(weights),
            b: Some(Tensor::from_vec(biases, (EXPERTS, HIDDEN), &device)?),
        })?))
    };
    // These SiLU outputs and their Q8_1 block sums are exact, isolating bias handling.
    let gate = projection([32.0, 64.0])?;
    let up = projection([3.0, 4.0])?;
    let down = projection([5.0, 6.0])?;
    let experts = RoutedExperts {
        gate: &*gate,
        up: &*up,
        down: &*down,
        num_experts: EXPERTS,
    };
    let expected =
        0.25 * (HIDDEN as f32 * 32.0 * 3.0 + 5.0) + 0.75 * (HIDDEN as f32 * 64.0 * 4.0 + 6.0);
    for tokens in [1, GROUPED_PREFILL_MIN_TOKENS] {
        let xs = Tensor::zeros((tokens, HIDDEN), DType::F32, &device)?;
        let indices = Tensor::from_vec(
            (0..tokens * EXPERTS)
                .map(|i| (i % EXPERTS) as u32)
                .collect(),
            (tokens, EXPERTS),
            &device,
        )?;
        let routing = Tensor::from_vec(
            (0..tokens).flat_map(|_| [0.25f32, 0.75]).collect(),
            (tokens, EXPERTS),
            &device,
        )?;
        let result = experts.forward(
            RoutedInput {
                xs: &xs,
                indices: &indices,
                routing: &routing,
                output_dtype: DType::F32,
                lora: None,
            },
            RoutedActivation {
                fused: Some(GluActivationType::Silu),
                fallback: &candle_nn::Activation::Silu,
            },
        )?;
        let error = (result - f64::from(expected))?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error < 0.02, "tokens={tokens}, max error {error}");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn routed_stats_on_up_projection_disable_fusion() -> Result<()> {
    use crate::{AfqBits, AfqGroupSize, AfqLayer};

    let device = Device::new_cuda(0)?;
    let projection = || -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(AfqLayer::new(QuantMethodConfig::Afq {
            weight: Tensor::ones((EXPERTS, FEATURES, FEATURES), DType::F32, &device)?,
            bias: None,
            bits: AfqBits::Four,
            group_size: AfqGroupSize::Low,
        })?))
    };
    let gate = projection()?;
    let up = projection()?;
    let down = projection()?;
    up.begin_track_stats()?;
    let xs = Tensor::ones((1, FEATURES), DType::F32, &device)?;
    let indices = Tensor::new(&[[0u32, 1]], &device)?;
    let routing = Tensor::new(&[[0.5f32, 0.5]], &device)?;
    let experts = RoutedExperts {
        gate: &*gate,
        up: &*up,
        down: &*down,
        num_experts: EXPERTS,
    };
    experts.forward(
        RoutedInput {
            xs: &xs,
            indices: &indices,
            routing: &routing,
            output_dtype: DType::F32,
            lora: None,
        },
        RoutedActivation {
            fused: Some(GluActivationType::Silu),
            fallback: &candle_nn::Activation::Silu,
        },
    )?;
    assert_eq!(up.stats_snapshot(), Some((1, 2)));
    Ok(())
}


fn assert_tensors_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
    Ok(())
}

#[test]
fn cpu_weighted_reduce_matches_tensor_reference_across_dtypes_and_layouts() -> Result<()> {
    let device = Device::Cpu;
    let (tokens, topk, hidden) = (3, 4, 128);
    let values = (0..tokens * topk * hidden)
        .map(|index| ((index * 17 % 101) as f32 - 50.0) / 37.0)
        .collect::<Vec<_>>();
    let contiguous = Tensor::from_vec(values.clone(), (tokens, topk, hidden), &device)?;

    // Build the same logical [tokens, topk, hidden] tensor with a transposed
    // backing layout to exercise the generic helper's contiguous fallback.
    let mut transposed_values = vec![0f32; tokens * hidden * topk];
    for token in 0..tokens {
        for expert in 0..topk {
            for feature in 0..hidden {
                transposed_values[(token * hidden + feature) * topk + expert] =
                    values[(token * topk + expert) * hidden + feature];
            }
        }
    }
    let transposed = Tensor::from_vec(
        transposed_values,
        (tokens, hidden, topk),
        &device,
    )?
    .transpose(1, 2)?;

    let routing = Tensor::from_vec(
        (0..tokens * topk)
            .map(|index| (index % 7 + 1) as f32 / 16.0)
            .collect::<Vec<_>>(),
        (tokens, topk),
        &device,
    )?;

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for ys_f32 in [&contiguous, &transposed] {
            let ys = ys_f32.to_dtype(dtype)?;
            let expected = ys
                .to_dtype(DType::F32)?
                .broadcast_mul(&routing.unsqueeze(D::Minus1)?)?
                .sum(D::Minus2)?
                .to_dtype(dtype)?;
            let actual = topk_weight_reduce(&ys, &routing, dtype)?;

            assert_eq!(actual.dims(), &[tokens, hidden]);
            assert_eq!(actual.dtype(), dtype);
            let tolerance = match dtype {
                DType::F32 => 1e-5,
                DType::F16 => 1e-3,
                DType::BF16 => 5e-3,
                _ => unreachable!(),
            };
            assert_tensors_close(&actual, &expected, tolerance)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "manual generic CPU MoE weighted-reduction benchmark"]
fn cpu_weighted_reduce_benchmark() -> Result<()> {
    const WARMUP: usize = 5;
    const ITERATIONS: usize = 15;
    let device = Device::Cpu;

    for (label, tokens, topk, hidden) in [
        ("decode", 1usize, 4usize, 4096usize),
        ("prefill", 64usize, 4usize, 4096usize),
    ] {
        let ys = Tensor::from_vec(
            (0..tokens * topk * hidden)
                .map(|index| ((index * 13 % 97) as f32 - 48.0) / 31.0)
                .collect::<Vec<_>>(),
            (tokens, topk, hidden),
            &device,
        )?;
        let routing = Tensor::from_vec(
            (0..tokens * topk)
                .map(|index| (index % topk + 1) as f32 / 10.0)
                .collect::<Vec<_>>(),
            (tokens, topk),
            &device,
        )?;

        let baseline = || {
            ys.to_dtype(DType::F32)?
                .broadcast_mul(&routing.unsqueeze(D::Minus1)?)?
                .sum(D::Minus2)
        };
        let reference = baseline()?;
        let optimized = topk_weight_reduce(&ys, &routing, DType::F32)?;
        assert_tensors_close(&optimized, &reference, 1e-5)?;

        for _ in 0..WARMUP {
            std::hint::black_box(baseline()?);
            std::hint::black_box(topk_weight_reduce(&ys, &routing, DType::F32)?);
        }

        let mut baseline_ms = Vec::with_capacity(ITERATIONS);
        let mut optimized_ms = Vec::with_capacity(ITERATIONS);
        for _ in 0..ITERATIONS {
            let start = std::time::Instant::now();
            std::hint::black_box(baseline()?);
            baseline_ms.push(start.elapsed().as_secs_f64() * 1000.0);

            let start = std::time::Instant::now();
            std::hint::black_box(topk_weight_reduce(&ys, &routing, DType::F32)?);
            optimized_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        baseline_ms.sort_by(f64::total_cmp);
        optimized_ms.sort_by(f64::total_cmp);

        let baseline_median = baseline_ms[ITERATIONS / 2];
        let optimized_median = optimized_ms[ITERATIONS / 2];
        let intermediate_mib =
            (tokens * topk * hidden * std::mem::size_of::<f32>()) as f64
                / (1024.0 * 1024.0);
        eprintln!(
            "Generic CPU MoE weighted reduce ({label}): baseline_median_ms={baseline_median:.3}, optimized_median_ms={optimized_median:.3}, speedup={:.2}x, avoided_intermediate={intermediate_mib:.2}MiB",
            baseline_median / optimized_median,
        );
    }
    Ok(())
}
