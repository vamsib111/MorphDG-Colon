# ======================================================================
# AUTO-GENERATED FROZEN MORPHDG-COLON IMPLEMENTATION
#
# Source stage : 8B-1
# Parent spec  : Stage-8A3
#
# This module contains NO dataset, filesystem, validation, test,
# external-data, or training-loop access.
# ======================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


FROZEN_SPEC_SHA256 = "a3c92005e646580170c4a6129c63339f834ad428a43c51fad792c30157df2ca9"

NUM_CLASSES = 9
NUM_DOMAINS = 3

IMAGE_POOLED_DIM = 1280
IMAGE_FUSION_DIM = 256

MORPHOLOGY_INPUT_DIM = 10
MORPHOLOGY_EMBED_DIM = 64

DOMAIN_CONTEXT_INPUT_DIM = 9
DOMAIN_CONTEXT_EMBED_DIM = 32

JOINT_CONTEXT_DIM = 128

IMAGE_PROJECTION_DROPOUT = 0.20
MORPHOLOGY_BRANCH_DROPOUT = 0.10
DOMAIN_CONTEXT_BRANCH_DROPOUT = 0.10
JOINT_CONTEXT_DROPOUT = 0.15
DOMAIN_HEAD_DROPOUT = 0.20

MORPHOLOGY_SAMPLE_DROPOUT = 0.15

FINAL_CLASSIFICATION_WEIGHT = 1.00
BASELINE_ANCHOR_WEIGHT = 0.25
DOMAIN_ADVERSARIAL_WEIGHT = 0.05
MORPHOLOGY_ALIGNMENT_WEIGHT = 0.05
RESIDUAL_ENERGY_WEIGHT = 0.01

GRADIENT_REVERSAL_LAMBDA = 1.00

CORRECTION_GATE_INITIAL_BIAS = -1.3862943611198906

BASE_LEARNING_RATE = 3e-4
MINIMUM_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4

ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8


class _GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            -ctx.coefficient * grad_output,
            None,
        )


def gradient_reverse(
    x,
    coefficient=GRADIENT_REVERSAL_LAMBDA,
):

    return _GradientReversalFunction.apply(
        x,
        float(coefficient),
    )


class MorphDGColon(nn.Module):

    def __init__(
        self,
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
    ):

        super().__init__()

        # --------------------------------------------------------------
        # EfficientNet-B0 image encoder
        # --------------------------------------------------------------

        self.backbone = efficientnet_b0(
            weights=weights
        )

        # Remove the original ImageNet classifier parameters.
        self.backbone.classifier = nn.Identity()

        # Direct image-only prediction anchor.
        # Its initialization remains PyTorch nn.Linear default.
        self.baseline_classifier = nn.Linear(
            IMAGE_POOLED_DIM,
            NUM_CLASSES,
        )

        # --------------------------------------------------------------
        # Image projection for fusion
        # --------------------------------------------------------------

        self.image_projection = nn.Sequential(
            nn.LayerNorm(
                IMAGE_POOLED_DIM
            ),
            nn.Linear(
                IMAGE_POOLED_DIM,
                512,
            ),
            nn.GELU(),
            nn.Dropout(
                IMAGE_PROJECTION_DROPOUT
            ),
            nn.Linear(
                512,
                IMAGE_FUSION_DIM,
            ),
            nn.LayerNorm(
                IMAGE_FUSION_DIM
            ),
        )

        # --------------------------------------------------------------
        # Morphology encoder
        # --------------------------------------------------------------

        self.morphology_encoder = nn.Sequential(
            nn.LayerNorm(
                MORPHOLOGY_INPUT_DIM
            ),
            nn.Linear(
                MORPHOLOGY_INPUT_DIM,
                64,
            ),
            nn.GELU(),
            nn.Dropout(
                MORPHOLOGY_BRANCH_DROPOUT
            ),
            nn.Linear(
                64,
                MORPHOLOGY_EMBED_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                MORPHOLOGY_EMBED_DIM
            ),
        )

        # --------------------------------------------------------------
        # Continuous domain-context encoder
        # --------------------------------------------------------------

        self.domain_context_encoder = nn.Sequential(
            nn.LayerNorm(
                DOMAIN_CONTEXT_INPUT_DIM
            ),
            nn.Linear(
                DOMAIN_CONTEXT_INPUT_DIM,
                32,
            ),
            nn.GELU(),
            nn.Dropout(
                DOMAIN_CONTEXT_BRANCH_DROPOUT
            ),
            nn.Linear(
                32,
                DOMAIN_CONTEXT_EMBED_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                DOMAIN_CONTEXT_EMBED_DIM
            ),
        )

        # --------------------------------------------------------------
        # Joint morphology + domain context
        # --------------------------------------------------------------

        self.joint_context_encoder = nn.Sequential(
            nn.Linear(
                MORPHOLOGY_EMBED_DIM
                + DOMAIN_CONTEXT_EMBED_DIM,
                JOINT_CONTEXT_DIM,
            ),
            nn.GELU(),
            nn.Dropout(
                JOINT_CONTEXT_DROPOUT
            ),
            nn.LayerNorm(
                JOINT_CONTEXT_DIM
            ),
        )

        # --------------------------------------------------------------
        # Gated residual feature fusion
        # --------------------------------------------------------------

        self.context_projection = nn.Linear(
            JOINT_CONTEXT_DIM,
            IMAGE_FUSION_DIM,
        )

        self.feature_gate_linear = nn.Linear(
            IMAGE_FUSION_DIM
            + JOINT_CONTEXT_DIM,
            IMAGE_FUSION_DIM,
        )

        self.fusion_norm = nn.LayerNorm(
            IMAGE_FUSION_DIM
        )

        # --------------------------------------------------------------
        # Bounded residual logit correction
        # --------------------------------------------------------------

        self.delta_head = nn.Linear(
            IMAGE_FUSION_DIM,
            NUM_CLASSES,
        )

        self.correction_gate_linear = nn.Linear(
            JOINT_CONTEXT_DIM,
            1,
        )

        # --------------------------------------------------------------
        # Training-only domain-adversarial head
        # --------------------------------------------------------------

        self.domain_classifier = nn.Sequential(
            nn.Linear(
                IMAGE_FUSION_DIM,
                64,
            ),
            nn.GELU(),
            nn.Dropout(
                DOMAIN_HEAD_DROPOUT
            ),
            nn.Linear(
                64,
                NUM_DOMAINS,
            ),
        )

        # --------------------------------------------------------------
        # Training-only image-to-morphology alignment head
        # --------------------------------------------------------------

        self.alignment_head = nn.Sequential(
            nn.Linear(
                IMAGE_FUSION_DIM,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                MORPHOLOGY_EMBED_DIM,
            ),
            nn.LayerNorm(
                MORPHOLOGY_EMBED_DIM
            ),
        )

        self._initialize_new_modules()

    def _initialize_new_modules(self):

        # Baseline classifier intentionally keeps nn.Linear default
        # initialization to mirror the frozen baseline behavior.

        xavier_linear_modules = [
            self.image_projection[1],
            self.image_projection[4],

            self.morphology_encoder[1],
            self.morphology_encoder[4],

            self.domain_context_encoder[1],
            self.domain_context_encoder[4],

            self.joint_context_encoder[0],

            self.context_projection,
            self.feature_gate_linear,

            self.domain_classifier[0],
            self.domain_classifier[3],

            self.alignment_head[0],
            self.alignment_head[2],
        ]

        for module in xavier_linear_modules:

            nn.init.xavier_uniform_(
                module.weight
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        # LayerNorm defaults are explicitly enforced.
        for module in self.modules():

            if isinstance(
                module,
                nn.LayerNorm,
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

        # Exact frozen anchor-preserving initialization.
        nn.init.zeros_(
            self.delta_head.weight
        )

        nn.init.zeros_(
            self.delta_head.bias
        )

        nn.init.zeros_(
            self.correction_gate_linear.weight
        )

        nn.init.constant_(
            self.correction_gate_linear.bias,
            CORRECTION_GATE_INITIAL_BIAS,
        )

    def _encode_image(
        self,
        image,
    ):

        x = self.backbone.features(
            image
        )

        x = self.backbone.avgpool(
            x
        )

        x = torch.flatten(
            x,
            1,
        )

        if x.shape[1] != IMAGE_POOLED_DIM:

            raise RuntimeError(
                "Unexpected EfficientNet-B0 pooled dimension: "
                f"{tuple(x.shape)}"
            )

        return x

    def _apply_morphology_sample_dropout(
        self,
        morphology_embedding,
    ):

        if (
            not self.training
            or MORPHOLOGY_SAMPLE_DROPOUT <= 0.0
        ):

            return morphology_embedding

        keep_mask = (
            torch.rand(
                (
                    morphology_embedding.shape[0],
                    1,
                ),
                device=morphology_embedding.device,
            )
            >= MORPHOLOGY_SAMPLE_DROPOUT
        )

        keep_mask = keep_mask.to(
            dtype=morphology_embedding.dtype
        )

        return (
            morphology_embedding
            * keep_mask
        )

    def forward(
        self,
        image,
        morphology,
        domain_context,
        return_aux=True,
    ):

        if image.ndim != 4:
            raise ValueError(
                "image must have shape [B,3,H,W]"
            )

        if image.shape[1] != 3:
            raise ValueError(
                "image channel dimension must be 3"
            )

        if (
            morphology.ndim != 2
            or morphology.shape[1]
            != MORPHOLOGY_INPUT_DIM
        ):

            raise ValueError(
                "morphology must have shape [B,10]"
            )

        if (
            domain_context.ndim != 2
            or domain_context.shape[1]
            != DOMAIN_CONTEXT_INPUT_DIM
        ):

            raise ValueError(
                "domain_context must have shape [B,9]"
            )

        batch_size = image.shape[0]

        if (
            morphology.shape[0] != batch_size
            or domain_context.shape[0] != batch_size
        ):

            raise ValueError(
                "image, morphology and domain_context batch "
                "dimensions must match"
            )

        image_pooled = self._encode_image(
            image
        )

        baseline_logits = self.baseline_classifier(
            image_pooled
        )

        image_fusion_embedding = self.image_projection(
            image_pooled
        )

        morphology_embedding = self.morphology_encoder(
            morphology
        )

        domain_context_embedding = self.domain_context_encoder(
            domain_context
        )

        morphology_for_fusion = (
            self._apply_morphology_sample_dropout(
                morphology_embedding
            )
        )

        joint_context_embedding = (
            self.joint_context_encoder(
                torch.cat(
                    [
                        morphology_for_fusion,
                        domain_context_embedding,
                    ],
                    dim=1,
                )
            )
        )

        projected_context = self.context_projection(
            joint_context_embedding
        )

        feature_gate_vector = torch.sigmoid(
            self.feature_gate_linear(
                torch.cat(
                    [
                        image_fusion_embedding,
                        joint_context_embedding,
                    ],
                    dim=1,
                )
            )
        )

        fused_embedding = self.fusion_norm(
            image_fusion_embedding
            + feature_gate_vector
            * projected_context
        )

        delta_logits = self.delta_head(
            fused_embedding
        )

        correction_gate = torch.sigmoid(
            self.correction_gate_linear(
                joint_context_embedding
            )
        )

        final_logits = (
            baseline_logits
            + correction_gate
            * delta_logits
        )

        domain_logits = self.domain_classifier(
            gradient_reverse(
                image_fusion_embedding,
                GRADIENT_REVERSAL_LAMBDA,
            )
        )

        image_morphology_alignment_embedding = (
            self.alignment_head(
                image_fusion_embedding
            )
        )

        if not return_aux:
            return final_logits

        return {
            "final_logits":
                final_logits,

            "baseline_logits":
                baseline_logits,

            "delta_logits":
                delta_logits,

            "feature_gate_vector":
                feature_gate_vector,

            "correction_gate":
                correction_gate,

            "morphology_embedding":
                morphology_embedding,

            "domain_context_embedding":
                domain_context_embedding,

            "joint_context_embedding":
                joint_context_embedding,

            "image_fusion_embedding":
                image_fusion_embedding,

            "fused_embedding":
                fused_embedding,

            "domain_logits":
                domain_logits,

            "image_morphology_alignment_embedding":
                image_morphology_alignment_embedding,
        }

    @torch.no_grad()
    def inference(
        self,
        image,
        morphology,
        domain_context,
    ):

        return self.forward(
            image,
            morphology,
            domain_context,
            return_aux=False,
        )


def ensure_finite_loss_dict(
    losses,
):

    for name, value in losses.items():

        if not torch.isfinite(
            value
        ).all():

            raise FloatingPointError(
                f"Non-finite loss detected: {name}"
            )


def compute_morphdg_losses(
    outputs,
    class_labels,
    hard_domain_labels,
):

    final_ce = F.cross_entropy(
        outputs[
            "final_logits"
        ],
        class_labels,
    )

    baseline_ce = F.cross_entropy(
        outputs[
            "baseline_logits"
        ],
        class_labels,
    )

    domain_ce = F.cross_entropy(
        outputs[
            "domain_logits"
        ],
        hard_domain_labels,
    )

    image_alignment = F.normalize(
        outputs[
            "image_morphology_alignment_embedding"
        ],
        p=2,
        dim=1,
    )

    morphology_target = F.normalize(
        outputs[
            "morphology_embedding"
        ].detach(),
        p=2,
        dim=1,
    )

    morphology_alignment = torch.mean(
        1.0
        - torch.sum(
            image_alignment
            * morphology_target,
            dim=1,
        )
    )

    gated_delta = (
        outputs[
            "correction_gate"
        ]
        * outputs[
            "delta_logits"
        ]
    )

    residual_energy = torch.mean(
        gated_delta
        * gated_delta
    )

    total = (
        FINAL_CLASSIFICATION_WEIGHT
        * final_ce

        + BASELINE_ANCHOR_WEIGHT
        * baseline_ce

        + DOMAIN_ADVERSARIAL_WEIGHT
        * domain_ce

        + MORPHOLOGY_ALIGNMENT_WEIGHT
        * morphology_alignment

        + RESIDUAL_ENERGY_WEIGHT
        * residual_energy
    )

    losses = {
        "final_classification":
            final_ce,

        "baseline_anchor":
            baseline_ce,

        "domain_adversarial":
            domain_ce,

        "morphology_alignment":
            morphology_alignment,

        "residual_energy":
            residual_energy,

        "total":
            total,
    }

    ensure_finite_loss_dict(
        losses
    )

    return losses


def build_optimizer(
    model,
):

    layernorm_parameter_ids = set()

    for module in model.modules():

        if isinstance(
            module,
            nn.LayerNorm,
        ):

            for parameter in module.parameters(
                recurse=False
            ):

                layernorm_parameter_ids.add(
                    id(parameter)
                )

    decay_parameters = []
    no_decay_parameters = []

    assigned_parameter_ids = set()

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        parameter_id = id(
            parameter
        )

        if parameter_id in assigned_parameter_ids:

            raise RuntimeError(
                "Parameter assigned more than once to optimizer groups"
            )

        assigned_parameter_ids.add(
            parameter_id
        )

        if (
            name.endswith(
                ".bias"
            )
            or parameter_id
            in layernorm_parameter_ids
        ):

            no_decay_parameters.append(
                parameter
            )

        else:

            decay_parameters.append(
                parameter
            )

    expected_trainable_ids = {
        id(parameter)
        for parameter
        in model.parameters()
        if parameter.requires_grad
    }

    if assigned_parameter_ids != expected_trainable_ids:

        raise RuntimeError(
            "Optimizer parameter groups do not cover every "
            "trainable parameter exactly once"
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    decay_parameters,

                "weight_decay":
                    WEIGHT_DECAY,

                "lr":
                    BASE_LEARNING_RATE,
            },
            {
                "params":
                    no_decay_parameters,

                "weight_decay":
                    0.0,

                "lr":
                    BASE_LEARNING_RATE,
            },
        ],
        lr=BASE_LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
    )

    return optimizer


class WarmupCosineScheduler:

    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps,
        base_lr=BASE_LEARNING_RATE,
        minimum_lr=MINIMUM_LEARNING_RATE,
        warmup_start_factor=0.10,
    ):

        if total_steps <= 0:
            raise ValueError(
                "total_steps must be positive"
            )

        if warmup_steps < 0:
            raise ValueError(
                "warmup_steps must be non-negative"
            )

        if warmup_steps >= total_steps:
            raise ValueError(
                "warmup_steps must be less than total_steps"
            )

        self.optimizer = optimizer

        self.total_steps = int(
            total_steps
        )

        self.warmup_steps = int(
            warmup_steps
        )

        self.base_lr = float(
            base_lr
        )

        self.minimum_lr = float(
            minimum_lr
        )

        self.warmup_start_factor = float(
            warmup_start_factor
        )

        self.successful_steps = 0

        initial_lr = (
            self.base_lr
            * self.warmup_start_factor
            if self.warmup_steps > 0
            else self.base_lr
        )

        self._set_lr(
            initial_lr
        )

    def _set_lr(
        self,
        learning_rate,
    ):

        for group in self.optimizer.param_groups:
            group[
                "lr"
            ] = float(
                learning_rate
            )

    def _learning_rate_for_step(
        self,
        successful_step,
    ):

        successful_step = int(
            successful_step
        )

        if (
            self.warmup_steps > 0
            and successful_step
            <= self.warmup_steps
        ):

            progress = (
                successful_step
                / self.warmup_steps
            )

            factor = (
                self.warmup_start_factor
                + (
                    1.0
                    - self.warmup_start_factor
                )
                * progress
            )

            return (
                self.base_lr
                * factor
            )

        cosine_denominator = (
            self.total_steps
            - self.warmup_steps
        )

        cosine_progress = (
            successful_step
            - self.warmup_steps
        ) / cosine_denominator

        cosine_progress = min(
            1.0,
            max(
                0.0,
                cosine_progress,
            ),
        )

        cosine_value = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * cosine_progress
                )
            )
        )

        return (
            self.minimum_lr
            + (
                self.base_lr
                - self.minimum_lr
            )
            * cosine_value
        )

    def step(
        self,
    ):

        self.successful_steps += 1

        learning_rate = (
            self._learning_rate_for_step(
                self.successful_steps
            )
        )

        self._set_lr(
            learning_rate
        )

        return learning_rate

    def get_last_lr(
        self,
    ):

        return [
            float(
                group[
                    "lr"
                ]
            )
            for group
            in self.optimizer.param_groups
        ]

    def state_dict(
        self,
    ):

        return {
            "total_steps":
                self.total_steps,

            "warmup_steps":
                self.warmup_steps,

            "base_lr":
                self.base_lr,

            "minimum_lr":
                self.minimum_lr,

            "warmup_start_factor":
                self.warmup_start_factor,

            "successful_steps":
                self.successful_steps,

            "last_lr":
                self.get_last_lr(),
        }

    def load_state_dict(
        self,
        state,
    ):

        if int(
            state[
                "total_steps"
            ]
        ) != self.total_steps:

            raise ValueError(
                "Scheduler total_steps mismatch"
            )

        if int(
            state[
                "warmup_steps"
            ]
        ) != self.warmup_steps:

            raise ValueError(
                "Scheduler warmup_steps mismatch"
            )

        self.base_lr = float(
            state[
                "base_lr"
            ]
        )

        self.minimum_lr = float(
            state[
                "minimum_lr"
            ]
        )

        self.warmup_start_factor = float(
            state[
                "warmup_start_factor"
            ]
        )

        self.successful_steps = int(
            state[
                "successful_steps"
            ]
        )

        last_lr = state[
            "last_lr"
        ]

        if len(
            last_lr
        ) != len(
            self.optimizer.param_groups
        ):

            raise ValueError(
                "Scheduler optimizer-group count mismatch"
            )

        for group, learning_rate in zip(
            self.optimizer.param_groups,
            last_lr,
        ):

            group[
                "lr"
            ] = float(
                learning_rate
            )
