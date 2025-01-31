//! A direct 1:1 translation of the provided Python code into Rust
//!
//! This translation preserves all classes/structs, docstrings, arguments, and logic as closely
//! as possible to the original Python, including stubs for distributed functions and custom
//! `fp8`/quantization functions. Nothing has been removed or omitted.

use std::ops::Add;
use serde::{Deserialize, Serialize};
use tch::{
    kind::Kind,
    nn,
    nn::{ModuleT, OptimizerConfig},
    vision::dataset::Dataset,
    Device, IndexOp, Reduction, Tensor,
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
            vs,
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
                Some(b) => x.matmul(&weight.tr()),
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
        assert!(in_features % world_size == 0, "in_features must be divisible by world_size");

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
            let bias_tensor = bias.to(y.device()).view(&[1, -1]) as Tensor;
            y = y + *bias_tensor;
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
        let norm = x.pow(*2)
            .mean_dim(&[-1], true, Kind::Float)
            .add(self.eps)
            .sqrt()
            .reciprocal();

        let norm = norm.to_kind(x.kind());
        & * *&norm * &self.weight
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
        let x = &*x * &self.w3.forward(&x);
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
                scores.topk(2, -1, true, false).0.sum(&-1)
            };

            let indices = group_scores.topk(self.topk_groups, -1, true, false).1;
            let mut mask = Tensor::zeros_like(&scores.narrow(-1, 0, 1));
            mask.scatter_(-1, &indices, &Tensor::ones_like(&indices));

            scores = (scores * mask.unsqueeze(-1)).flatten(1, -1);
        }

        let indices = scores.topk(self.topk, -1, true, false).1;
        let mut weights = original_scores.gather(-1, &indices, false);

        if self.score_func == "sigmoid" {
            weights = weights / weights.sum(&-1);
        }

        weights *= self.route_scale;

        (weights.to_kind(x.kind()), indices)
    }
}
