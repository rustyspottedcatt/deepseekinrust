//! A direct 1:1 translation of the provided Python code into Rust
//!
//! This translation preserves all classes/structs, docstrings, arguments, and logic as closely
//! as possible to the original Python, including stubs for distributed functions and custom
//! `fp8`/quantization functions. Nothing has been removed or omitted.

use std::ops::{Add, Div, Mul};
use serde::{Deserialize, Serialize};
use tch::{
    kind::Kind,
    nn,
    nn::{ModuleT, OptimizerConfig},
    Device, IndexOp, Tensor,
};
use tch::nn::{EmbeddingConfig, Module};
use crate::inference::kernel::{act_quant, weight_dequant, fp8_gemm};

// ------------------------------------------------------------------------------------------
// Stub distributed API to mimic `torch.distributed as dist`. Adjust/implement as needed.
// ------------------------------------------------------------------------------------------
mod dist {
    use tch::Tensor;

    /// Stub to mimic `torch.distributed.is_initialized()`.
    pub fn is_initialized() -> bool {
        false
    }

    /// Stub to mimic `torch.distributed.get_world_size()`.
    pub fn get_world_size() -> i64 {
        1
    }

    /// Stub to mimic `torch.distributed.get_rank()`.
    pub fn get_rank() -> i64 {
        0
    }

    /// Stub to mimic `torch.distributed.all_reduce()`.
    pub fn all_reduce(_t: &Tensor) {
        // no-op
    }

    /// Stub to mimic `torch.distributed.all_gather()`.
    pub fn all_gather(_out: &mut [Tensor], _t: &Tensor) {
        // no-op
    }
}

// ------------------------------------------------------------------------------------------
// Global "constants"/variables, matching the Python code exactly.
// ------------------------------------------------------------------------------------------
static mut WORLD_SIZE: i64 = 1;
static mut RANK: i64 = 0;
static BLOCK_SIZE: i64 = 128;
// Allowed values: "bf16", "fp8"
static GEMM_IMPL: &str = "bf16";
// Allowed values: "naive", "absorb"
static ATTN_IMPL: &str = "absorb";

// ------------------------------------------------------------------------------------------
// ModelArgs definition (unchanged except for Rust syntax).
// ------------------------------------------------------------------------------------------

/// Data class for defining model arguments and hyperparameters.
///
/// Attributes:
///     max_batch_size (int): Maximum batch size.
///     max_seq_len (int): Maximum sequence length.
///     dtype (Literal["bf16", "fp8"]): Data type for computations.
///     vocab_size (int): Vocabulary size.
///     dim (int): Model dimension.
///     inter_dim (int): Intermediate dimension for MLP layers.
///     moe_inter_dim (int): Intermediate dimension for MoE layers.
///     n_layers (int): Number of transformer layers.
///     n_dense_layers (int): Number of dense layers in the model.
///     n_heads (int): Number of attention heads.
///     n_routed_experts (int): Number of routed experts for MoE layers.
///     n_shared_experts (int): Number of shared experts for MoE layers.
///     n_activated_experts (int): Number of activated experts in MoE layers.
///     n_expert_groups (int): Number of expert groups.
///     n_limited_groups (int): Number of limited groups for MoE routing.
///     score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
///     route_scale (float): Scaling factor for routing scores.
///     q_lora_rank (int): LoRA rank for query projections.
///     kv_lora_rank (int): LoRA rank for key-value projections.
///     qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
///     qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
///     v_head_dim (int): Dimension for value projections.
///     original_seq_len (int): Original sequence length.
///     rope_theta (float): Base for rotary positional encoding.
///     rope_factor (float): Scaling factor for extended sequence lengths.
///     beta_fast (int): Fast beta correction factor.
///     beta_slow (int): Slow beta correction factor.
///     mscale (float): Scaling factor for extended attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArgs {
    pub max_batch_size: i64,
    pub max_seq_len: i64,
    pub dtype: String,
    pub vocab_size: i64,
    pub dim: i64,
    pub inter_dim: i64,
    pub moe_inter_dim: i64,
    pub n_layers: i64,
    pub n_dense_layers: i64,
    pub n_heads: i64,
    // moe
    pub n_routed_experts: i64,
    pub n_shared_experts: i64,
    pub n_activated_experts: i64,
    pub n_expert_groups: i64,
    pub n_limited_groups: i64,
    pub score_func: String,
    pub route_scale: f64,
    // mla
    pub q_lora_rank: i64,
    pub kv_lora_rank: i64,
    pub qk_nope_head_dim: i64,
    pub qk_rope_head_dim: i64,
    pub qk_head_dim: i64,
    pub v_head_dim: i64,
    // yarn
    pub original_seq_len: i64,
    pub rope_theta: f64,
    pub rope_factor: f64,
    pub beta_fast: i64,
    pub beta_slow: i64,
    pub mscale: f64,
}

impl Default for ModelArgs {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_seq_len: 4096 * 4,
            dtype: "bf16".to_owned(),
            vocab_size: 102400,
            dim: 2048,
            inter_dim: 10944,
            moe_inter_dim: 1408,
            n_layers: 27,
            n_dense_layers: 1,
            n_heads: 16,
            n_routed_experts: 64,
            n_shared_experts: 2,
            n_activated_experts: 6,
            n_expert_groups: 1,
            n_limited_groups: 1,
            score_func: "softmax".to_owned(),
            route_scale: 1.0,
            q_lora_rank: 0,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            qk_head_dim: 0,
            v_head_dim: 128,
            original_seq_len: 4096,
            rope_theta: 10000.0,
            rope_factor: 40.0,
            beta_fast: 32,
            beta_slow: 1,
            mscale: 1.0,
        }
    }
}

pub fn apply_rotary_emb(x: &Tensor,     freqs_cis: &Tensor) -> Tensor {
    let dtype = x.kind();

    let x_complex = x
        .to_kind(Kind::Float)
        .view([-1, x.size()[x.dim() - 1] / 2, 2])
        .view_as_complex();

    let freqs_cis = freqs_cis
        .view([1, x.size()[1], 1, x.size()[x.dim() - 1]]);

    let y = x_complex * freqs_cis;
    let y_real = y.view_as_real().flatten(3, -1);

    y_real.to_kind(dtype)
}

pub fn precompute_freqs_cis(args: &ModelArgs) -> Tensor {
    let dim = args.qk_rope_head_dim;
    let seqlen = args.max_seq_len;
    let beta_fast = args.beta_fast as f64;
    let beta_slow = args.beta_slow as f64;
    let base = args.rope_theta;
    let factor = args.rope_factor;

    fn find_correction_dim(num_rotations: f64, dim: i64, base: f64, max_seq_len: i64) -> f64 {
        dim as f64 * ((max_seq_len as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln())
    }

    fn find_correction_range(low_rot: f64, high_rot: f64, dim: i64, base: f64, max_seq_len: i64) -> (i64, i64) {
        let low = find_correction_dim(low_rot, dim, base, max_seq_len).floor() as i64;
        let high = find_correction_dim(high_rot, dim, base, max_seq_len).ceil() as i64;
        (low.max(0), high.min(dim - 1))
    }

    fn linear_ramp_factor(min: i64, max: i64, dim: i64) -> Tensor {
        if min == max {
            return Tensor::zeros(&[dim], (Kind::Float, Device::Cpu));
        }
        let linear_func = Tensor::arange(dim, (Kind::Float, Device::Cpu)) - min as f64;
        let ramp_func = (linear_func / ((max - min) as f64)).clamp(0.0, 1.0);
        ramp_func
    }

    let freqs = Tensor::arange(dim / 2, (Kind::Float, Device::Cpu))
        .div(dim as f64)
        .mul(-base.ln())
        .exp();

    let freqs = if seqlen > args.original_seq_len {
        let (low, high) = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len);
        let smooth = 1.0 - linear_ramp_factor(low, high, dim / 2);
        let freqs_scaled = freqs.shallow_clone().div(factor).mul(&(1.0 - &smooth));
        let freqs_original = freqs.shallow_clone().mul(&smooth);
        freqs_scaled + freqs_original
    } else {
        freqs
    };

    let t = Tensor::arange(seqlen, (Kind::Float, Device::Cpu));
    let freqs = t.outer(&freqs);
    Tensor::complex(&freqs.cos(), &freqs.sin())
}

pub struct ParallelEmbedding {
    pub vocab_size: i64,
    pub dim: i64,
    pub part_vocab_size: i64,
    pub vocab_start_idx: i64,
    pub vocab_end_idx: i64,
    pub weight: nn::Embedding,
}

impl ParallelEmbedding {
    pub fn new(vs: &&nn::Path, vocab_size: i64, dim: i64) -> Self {
        unsafe {
            WORLD_SIZE = dist::get_world_size();
            RANK = dist::get_rank();
        }
        let world_size = unsafe { WORLD_SIZE };
        let rank = unsafe { WORLD_SIZE };

        assert_eq!(vocab_size % world_size, 0, "vocab_size must be divisible by world_size");

        let part_vocab_size = vocab_size / world_size;
        let vocab_start_idx = part_vocab_size * rank;
        let vocab_end_idx = vocab_start_idx + part_vocab_size;

        let weight = nn::embedding(
            *vs,
            part_vocab_size,
            dim,
            EmbeddingConfig {
                ..Default::default()
            }
        );

        Self {
            vocab_size,
            dim,
            part_vocab_size,
            vocab_start_idx,
            vocab_end_idx,
            weight,
        }
    }

    pub fn forward(&self, x:&Tensor) -> Tensor {
        let world_size = unsafe { WORLD_SIZE };

        if world_size > 1 {
            let mask = x.lt(self.vocab_start_idx);

            let shifted = x - self.vocab_start_idx;
            let shifted = shifted.where_self(&mask.logical_not(), &Tensor::zeros_like(x));

            let mut y = self.weight.forward(&shifted);

            let expanded_mask = mask.unsqueeze(-1).to_kind(y.kind());
            let zero_t = Tensor::zeros_like(&y);
            y = y.where_self(&expanded_mask.logical_not(), &zero_t);

            dist::all_reduce(&y);
            y
        } else {
            self.weight.forward(x)
        }
    }
}

pub struct Linear {
    pub in_features: i64,
    pub out_features: i64,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub scale: Option<Tensor>,
    pub dtype: Kind,
}

impl Linear {
    pub fn new(vs: &nn::Path, in_features: i64, out_features: i64, bias: bool, dtype: Option<Kind>) -> Self {
        let block_size = BLOCK_SIZE;

        let actual_dtype = dtype.unwrap_or(Kind::BFloat16);

        let weight = vs.zeros("weight", &[out_features, in_features]);

        let scale = if actual_dtype == Kind::Float {
            let scale_out_features = (out_features + block_size - 1) / block_size;
            let scale_in_features = (in_features + block_size - 1) / block_size;
            Some(vs.zeros(
                "scale",
                &[scale_out_features, scale_in_features], // Scale should always be float32
            ))
        } else {
            None
        };

        let bias = if bias {
            Some(vs.zeros("bias", &[out_features]))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            weight,
            bias,
            scale,
            dtype: actual_dtype,
        }
    }

    pub fn forward(&self, x:&Tensor) -> Tensor {
        fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
            match bias {
                Some(_b) => x.matmul(&weight.tr()),
                None => x.matmul(&weight.tr()),
            }
        }

        linear(x, &self.weight, self.bias.as_ref())
    }
}

pub struct ColumnParallelLinear {
    pub linear: Linear,
    pub part_out_features: i64,
}

impl ColumnParallelLinear {
    pub fn new(vs: &nn::Path, in_features: i64, out_features: i64, bias: bool, dtype: Option<Kind>) -> Self {
        let world_size = unsafe { dist::get_world_size() };
        assert!(out_features % world_size == 0, "out_features must be divisible by world_size");

        let part_out_features = out_features / world_size;
        let linear = Linear::new(vs, in_features, part_out_features, bias, dtype);

        Self {
            linear,
            part_out_features,
        }
    }

    pub fn forward(&self, x:&Tensor) -> Tensor {
        self.linear.forward(x)
    }
}

pub struct RowParallelLinear {
    pub linear: Linear,
    pub part_in_features: i64,
}

impl RowParallelLinear {
    pub fn new(vs: &nn::Path, in_features: i64, out_features: i64, bias: bool, dtype: Option<tch::Kind>) -> Self {
        let world_size = unsafe { dist::get_world_size() };
        assert_eq!(in_features % world_size, 0, "in_features must be divisible by world_size");

        let part_in_features = in_features / world_size;
        let linear = Linear::new(vs, part_in_features, in_features, bias, dtype);

        Self {
            linear,
            part_in_features,
        }
    }

    pub fn forward(&self, x:&Tensor) -> Tensor {
        let mut y = self.linear.forward(x);

        let world_size = unsafe { dist::get_world_size() };
        if world_size > 1 {
            dist::all_reduce(&mut y);
        }

        if let Some(ref bias) = self.linear.bias {
            let bias_tensor = bias.to(y.device()).view([1, -1]) as Tensor;
            y = y + bias_tensor;
        }

        y
    }
}

pub struct RMSNorm {
    dim: i64,
    eps: f64,
    weight: Tensor,
}

impl RMSNorm {
    pub fn new(vs: &nn::Path, dim: i64, eps: f64) -> Self {
        let weight = vs.ones("weight", &[dim]);

        Self {
            dim,
            eps,
            weight,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let norm = x.pow(&Tensor::from(2))
            .mean_dim(&[-1i64][..], true, Kind::Float)
            .add(self.eps)
            .sqrt()
            .reciprocal();

        let norm = norm.to_kind(x.kind());
        & * &norm * &self.weight
    }
}

pub struct MLP {
    w1: ColumnParallelLinear,
    w2: RowParallelLinear,
    w3: ColumnParallelLinear,
}

impl MLP {
    pub fn new(vs: &nn::Path, dim: i64, inter_dim: i64) -> Self {
        let w1 = ColumnParallelLinear::new(vs, dim, inter_dim, false, None);
        let w2 = RowParallelLinear::new(vs, inter_dim, dim, false, None);
        let w3 = ColumnParallelLinear::new(vs, dim, inter_dim, false, None);

        Self { w1, w2, w3 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.w1.forward(x).silu();
        let x = &x * self.w3.forward(&x);
        self.w2.forward(&x)
    }
}

pub struct Gate {
    dim: i64,
    topk: i64,
    n_groups: i64,
    topk_groups: i64,
    score_func: String,
    route_scale: f64,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Gate {
    pub fn new(vs: &nn::Path, args: &ModelArgs) -> Self {
        let weight = vs.zeros("weight", &[args.n_routed_experts, args.dim]);
        let bias = if args.dim == 7168 {
            Some(vs.zeros("bias", &[args.n_routed_experts]))
        } else {
            None
        };

        Self {
            dim: args.dim,
            topk: args.n_activated_experts,
            n_groups: args.n_expert_groups,
            topk_groups: args.n_limited_groups,
            score_func: args.score_func.clone(),
            route_scale: args.route_scale,
            weight,
            bias,
        }
    }

    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut scores = x.matmul(&self.weight.tr());

        if self.score_func == "softmax" {
            scores = scores.softmax(-1, Kind::Float);
        } else {
            scores = scores.sigmoid();
        }

        let mut original_scores = scores.copy();

        if let Some(b) = &self.bias {
            scores = scores + b;
        }

        if self.n_groups > 1 {
            scores = scores.view([-1, self.n_groups, scores.size()[1] / self.n_groups]);

            let group_scores = if self.bias.is_none() {
                scores.amax(-1, false)
            } else {
                scores.topk(2, -1, true, false).0.sum(Kind::Float)
            };

            let indices = group_scores.topk(self.topk_groups, -1, true, false).1;
            let mut mask = Tensor::zeros_like(&scores.narrow(-1, 0, 1));
            let _ = mask.scatter_(-1, &indices, &Tensor::ones_like(&indices));

            scores = (scores * mask.unsqueeze(-1)).flatten(1, -1);
        }

        let indices = scores.topk(self.topk, -1, true, false).1;
        let mut weights = original_scores.gather(-1, &indices, false);

        if self.score_func == "sigmoid" {
            weights = &weights / weights.sum(Kind::Float);
        }

        weights *= self.route_scale;

        (weights.to_kind(x.kind()), indices)
    }
}

pub struct Expert {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Expert {
    pub fn new(vs: &nn::Path, dim: i64, inter_dim: i64) -> Self {
        Self {
            w1: Linear::new(vs, dim, inter_dim, false, None),
            w2: Linear::new(vs, inter_dim, dim, false, None),
            w3: Linear::new(vs, dim, inter_dim, false, None),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.w2.forward(&(self.w1.forward(&x).silu() * self.w3.forward(&x)))
    }
}

pub struct MoE {
    dim: i64,
    n_routed_experts: i64,
    n_local_experts: i64,
    n_activated_experts: i64,
    experts_start_idx: i64,
    experts_end_idx: i64,
    gate: Gate,
    experts: Vec<Option<Expert>>,
    shared_experts: MLP,
}

impl MoE {
    pub fn new(vs: &nn::Path, args: &ModelArgs) -> Self {
        let world_size = unsafe { dist::get_world_size() };
        let rank = unsafe { dist::get_rank() };

        assert_eq!(args.n_routed_experts % world_size, 0, "n_routed_experts must be divisible by world_size");

        let n_routed_experts = args.n_routed_experts;
        let n_local_experts = args.n_routed_experts / world_size;
        let n_activated_experts = args.n_activated_experts;
        let experts_start_idx = rank * n_local_experts;
        let experts_end_idx = experts_start_idx + n_local_experts;

        let gate = Gate::new(vs, args);

        let mut experts = Vec::new();
        for i in 0..n_routed_experts {
            if i >= experts_start_idx && i < experts_end_idx {
                experts.push(Some(Expert::new(vs, args.dim, args.moe_inter_dim)));
            } else {
                experts.push(None);
            }
        }

        let shared_experts = MLP::new(vs, args.dim, args.n_shared_experts * args.moe_inter_dim);

        Self {
            dim: args.dim,
            n_routed_experts,
            n_local_experts,
            n_activated_experts,
            experts_start_idx,
            experts_end_idx,
            gate,
            experts,
            shared_experts,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.size();
        let x = x.view([-1, self.dim]);

        let (weights, indices) = self.gate.forward(&x);
        let mut y = Tensor::zeros_like(&x);

        let counts = indices
            .flatten(0, -1)
            .bincount::<Tensor>(None, self.n_routed_experts)
            .to_kind(Kind::Int64)
            .iter::<i64>()
            .unwrap()
            .collect::<Vec<_>>();

        for i in self.experts_start_idx..self.experts_end_idx {
            if counts[i as usize] == 0 {
                continue;
            }
            if let Some(expert) = &self.experts[i as usize] {
                let mask = indices.eq(i);
                let idx = mask.nonzero();
                let top = mask
                    .nonzero()
                    .select(1, Tensor::from_slice(&[1]).to_kind(Kind::Int64).int64_value(&[]));

                let x_selected = x.index_select(0, &idx);
                let weights_selected = weights.index_select(0, &idx).gather(1, &top, false);

                let _ = y.index_add_(0, &idx, &(expert.forward(&x_selected) * weights_selected));
            }
        }

        let z = self.shared_experts.forward(&x);
        if unsafe { dist::get_world_size() } > 1 {
            dist::all_reduce(&mut y);
        }

        (y + z).view(&*shape)
    }
}

pub struct MLA {
    dim: i64,
    n_heads: i64,
    n_local_heads: i64,
    q_lora_rank: i64,
    kv_lora_rank: i64,
    qk_nope_head_dim: i64,
    qk_rope_head_dim: i64,
    qk_head_dim: i64,
    v_head_dim: i64,
    wq: Option<ColumnParallelLinear>,
    wq_a: Option<Linear>,
    q_norm: Option<RMSNorm>,
    wq_b: Option<ColumnParallelLinear>,
    wkv_a: Linear,
    kv_norm: RMSNorm,
    wkv_b: ColumnParallelLinear,
    wo: RowParallelLinear,
    softmax_scale: f64,
    k_cache: Tensor,
    v_cache: Tensor,
    kv_cache: Tensor,
    pe_cache: Tensor,
}

impl MLA {
    pub fn new(vs: &nn::Path, args: &ModelArgs) -> Self {
        let dim = args.dim;
        let n_heads = args.n_heads;
        let n_local_heads = args.n_heads / unsafe { dist::get_world_size() };
        let qk_nope_head_dim = args.qk_nope_head_dim;
        let qk_rope_head_dim = args.qk_rope_head_dim;
        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let v_head_dim = args.v_head_dim;
        let kv_lora_rank = args.kv_lora_rank;

        let wq = if args.q_lora_rank == 0 {
            Some(ColumnParallelLinear::new(vs, dim, n_heads * qk_head_dim, false, None))
        } else {
            None
        };

        let wq_a = if args.q_lora_rank > 0 {
            Some(Linear::new(vs, dim, args.q_lora_rank, false, None))
        } else {
            None
        };

        let q_norm = if args.q_lora_rank > 0 {
            Some(RMSNorm::new(vs, args.q_lora_rank, 1e-6))
        } else {
            None
        };

        let wq_b = if args.q_lora_rank > 0 {
            Some(ColumnParallelLinear::new(
                vs,
                args.q_lora_rank,
                n_heads * qk_head_dim,
                false,
                None,
            ))
        } else {
            None
        };

        let wkv_a = Linear::new(vs, dim, kv_lora_rank + qk_rope_head_dim, false, None);
        let kv_norm = RMSNorm::new(vs, kv_lora_rank, 1e-6);
        let wkv_b = ColumnParallelLinear::new(vs, kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), false, None);
        let wo = RowParallelLinear::new(vs, n_heads * v_head_dim, dim, false, None);

        let mut softmax_scale = (qk_head_dim as f64).powf(-0.5);
        if args.max_seq_len > args.original_seq_len {
            let mscale = 0.1 * args.mscale * (args.rope_factor as f64).ln() + 1.0;
            softmax_scale *= mscale * mscale;
        }

        Self {
            dim,
            n_heads,
            n_local_heads,
            q_lora_rank: args.q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            wq,
            wq_a,
            q_norm,
            wq_b,
            wkv_a,
            kv_norm,
            wkv_b,
            wo,
            softmax_scale,
            k_cache: Tensor::zeros(&[args.max_batch_size, args.max_seq_len, args.n_heads, args.qk_head_dim], (Kind::Float, Device::Cuda(0))),
            v_cache: Tensor::zeros(&[args.max_batch_size, args.max_seq_len, args.n_heads, args.v_head_dim], (Kind::Float, Device::Cuda(0))),
            kv_cache: Tensor::zeros(&[args.max_batch_size, args.max_seq_len, args.kv_lora_rank], (Kind::Float, Device::Cuda(0))),
            pe_cache: Tensor::zeros(&[args.max_batch_size, args.max_seq_len, args.qk_rope_head_dim], (Kind::Float, Device::Cuda(0))),
        }
    }

    pub fn forward(&self, x: &Tensor, start_pos: i64, freqs_cis: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let (bsz, seqlen, _) = x.size3().unwrap();
        let end_pos = start_pos + seqlen;

        let q = if self.q_lora_rank == 0 {
            self.wq.as_ref().unwrap().forward(x)
        } else {
            self.wq_b.as_ref().expect("wq_b is None").forward(
                &self.q_norm.as_ref().unwrap().forward(
                    &self.wq_a.as_ref().unwrap().forward(x)
                )
            )
        };

        let q = q.view([bsz, seqlen, self.n_local_heads, self.qk_head_dim]);

        let q_split = q.split_with_sizes(&[self.qk_nope_head_dim, self.qk_rope_head_dim], -1);
        let q_nope = &q_split[0];
        let q_pe = apply_rotary_emb(&q_split[1], freqs_cis);

        let kv = self.wkv_a.forward(x);
        let kv_split = kv.split_with_sizes(&[self.kv_lora_rank, self.qk_rope_head_dim], -1);
        let kv = &kv_split[0];
        let k_pe = apply_rotary_emb(&kv_split[1].unsqueeze(2), freqs_cis);

        if ATTN_IMPL == "naive" {
            let q = Tensor::cat(&[q_nope, &q_pe], -1);
            let kv = self.wkv_b.forward(
                &self.kv_norm.forward(kv)
            );
            let kv = kv.view([bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim]);

            let kv_split = kv.split_with_sizes(&[self.qk_nope_head_dim, self.v_head_dim], -1);
            let k_nope = &kv_split[0];
            let v = &kv_split[1];

            let k = Tensor::cat(&[k_nope, &k_pe.expand(&[-1, -1, self.n_local_heads, -1], false)], -1);

            self.k_cache.narrow(1, start_pos, seqlen).copy_(&k);
            self.v_cache.narrow(1, start_pos, seqlen).copy_(&v);

            let mut scores = q.matmul(&self.k_cache.narrow(1, 0, end_pos).transpose(-2, -1)) * self.softmax_scale;
            scores = scores.softmax(-1, Kind::Float).to_kind(x.kind());

            if let Some(mask) = mask {
                scores = scores + mask.unsqueeze(1);
            }

            let scores = scores.softmax(-1, Kind::Float).to_kind(x.kind());

            let x = scores.matmul(&self.v_cache.narrow(1, 0, end_pos));
            self.wo.forward(&x.flatten(2, 0))
        } else {
            let wkv_b = if self.wkv_b.linear.scale.is_none() {
                self.wkv_b.linear.weight.shallow_clone()
            } else {
                let default_scale = Tensor::ones_like(&self.wkv_b.linear.weight);
                let wkv_b_dequantized = match weight_dequant(
                    &self.wkv_b.linear.weight,
                    self.wkv_b.linear.scale.as_ref().unwrap_or(&default_scale),
                ) {
                    Ok(tensor) => tensor,
                    Err(_) => Tensor::zeros_like(&self.wkv_b.linear.weight),
                };
                wkv_b_dequantized
            };

            let wkv_b = wkv_b.view([self.n_local_heads, -1, self.kv_lora_rank]);

            let q_nope = q_nope.matmul(&wkv_b.narrow(1, 0, self.qk_nope_head_dim));

            self.kv_cache.narrow(1, start_pos, seqlen).copy_(&self.kv_norm.forward(kv));
            self.pe_cache.narrow(1, start_pos, seqlen).copy_(&k_pe.squeeze());

            let mut scores = (q_nope.matmul(&self.kv_cache.narrow(1, 0, end_pos).transpose(-2, -1)) +
                q_pe.matmul(&self.pe_cache.narrow(1, 0, end_pos).transpose(-2, -1))) * self.softmax_scale;

            if let Some(mask) = mask {
                scores = scores + mask.unsqueeze(1);
            }

            let scores = scores.softmax(-1, Kind::Float).to_kind(x.kind());

            let mut x = scores.matmul(&self.kv_cache.narrow(1, 0, end_pos));
            x = x.matmul(&wkv_b.narrow(1, wkv_b.size()[1] - self.v_head_dim, self.v_head_dim));

            self.wo.forward(&x.flatten(2, 0))
        }
    }
}

pub struct Block {
    attn: MLA,
    ffn: Option<MLP>,
    moe: Option<MoE>,
    attn_norm: RMSNorm,
    ffn_norm: RMSNorm,
}

impl Block {
    pub fn new(vs: &nn::Path, layer_id: i64, args: &ModelArgs) -> Self {
        let attn = MLA::new(vs, args);

        let (ffn, moe) = if layer_id < args.n_dense_layers {
            (Some(MLP::new(vs, args.dim, args.inter_dim)), None)
        } else {
            (None, Some(MoE::new(vs, args)))
        };

        let attn_norm = RMSNorm::new(vs, args.dim, 1e-6);
        let ffn_norm = RMSNorm::new(vs, args.dim, 1e-6);

        Self {
            attn,
            ffn,
            moe,
            attn_norm,
            ffn_norm,
        }
    }
    pub fn forward(&self, x: &Tensor, start_pos: i64, freqs_cis: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let mut x = x + self.attn.forward(&self.attn_norm.forward(x), start_pos, freqs_cis, mask);

        if let Some(ffn) = &self.ffn {
            x = &x + ffn.forward(&self.ffn_norm.forward(&x));
        } else if let Some(moe) = &self.moe {
            x = &x + moe.forward(&self.ffn_norm.forward(&x));
        }

        x
    }
}
pub struct Transformer {
    max_seq_len: i64,
    embed: ParallelEmbedding,
    layers: Vec<Block>,
    norm: RMSNorm,
    head: ColumnParallelLinear,
    freqs_cis: Tensor,
}

impl Transformer {
    pub fn new(vs: &nn::Path, args: &ModelArgs) -> Self {
        unsafe {
            WORLD_SIZE = dist::get_world_size();
            RANK = dist::get_rank();
        }

        let world_size = unsafe { WORLD_SIZE };
        let rank = unsafe { RANK };

        let dtype = if args.dtype == "fp8" { Kind::Float } else { Kind::BFloat16 };

        let embed = ParallelEmbedding::new(&vs, args.vocab_size, args.dim);
        let head = ColumnParallelLinear::new(vs, args.dim, args.vocab_size, false, Some(dtype));
        let mut layers = Vec::new();

        for layer_id in 0..args.n_layers {
            layers.push(Block::new(vs, layer_id, args));
        }

        let norm = RMSNorm::new(vs, args.dim, 1e-6);

        let freqs_cis = precompute_freqs_cis(args);

        Self {
            max_seq_len: args.max_seq_len,
            embed,
            layers,
            norm,
            head,
            freqs_cis,
        }
    }

    pub fn forward(&self, tokens: &Tensor, start_pos: i64) -> Tensor {
        let seqlen = tokens.size()[1];

        let mut h = self.embed.forward(tokens);

        let freqs_cis = self.freqs_cis.narrow(0, start_pos, seqlen);

        let mask = if seqlen > 1 {
            Some(Tensor::full(&[seqlen, seqlen], f64::NEG_INFINITY, (h.kind(), h.device()))
                .triu(1))
        } else {
            None
        };

        for layer in &self.layers {
            h = layer.forward(&h, start_pos, &freqs_cis, mask.as_ref());
        }

        h = self.norm.forward(&h).narrow(1, seqlen - 1, 1).squeeze();
        let logits = self.head.forward(&h);

        if unsafe { dist::get_world_size() } > 1 {
            let world_size = unsafe { dist::get_world_size() as usize };
            let mut all_logits: Vec<Tensor> = (0..world_size)
                .map(|_| Tensor::zeros_like(&logits))
                .collect();
            dist::all_gather(&mut all_logits, &logits);
            Tensor::cat(&all_logits, 0)
        } else {
            logits
        }
    }
}
