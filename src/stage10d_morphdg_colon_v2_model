"""
MorphDG-Colon V2
Reliability-Aware Stain-Robust Morphology-Gated Residual Fusion

Prospectively frozen by Stage10-C.

This module contains only model and loss definitions.
No dataset access, image decoding, external-data logic, model selection,
test-time augmentation, or test-time adaptation is implemented here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage8b1_morphdg_colon_model import (
    MorphDGColon,
    WarmupCosineScheduler,
    build_optimizer,
    compute_morphdg_losses,
    ensure_finite_loss_dict,
)


FROZEN_STAGE10C_SHA256 = (
    "f4b234feb488982cec9047e64557d1e6"
    "dae6353c9c1228feab33d9bb69b58ac8"
)

V2_METHOD_NAME = "MorphDG-Colon V2"
V2_NUM_CLASSES = 9

RELIABILITY_VECTOR_DIM = 3
RELIABILITY_EMBED_DIM = 16
JOINT_CONTEXT_DIM = 128
RELIABILITY_GATE_HIDDEN_DIM = 64

RELIABILITY_GATE_DROPOUT = 0.10
RELIABILITY_GATE_INITIAL_BIAS = -1.3862943611198906

CLASS_EVIDENCE_PRESERVATION_WEIGHT = 0.05

ENTROPY_EPSILON = 1e-8


class MorphDGColonV2(MorphDGColon):

    def __init__(
        self,
        weights=None,
    ):

        super().__init__(
            weights=weights
        )

        # V1 context-only correction gate is retained as inherited provenance
        # but is not used for V2 prediction.
        for parameter in self.correction_gate_linear.parameters():

            parameter.requires_grad_(
                False
            )

        self.reliability_encoder = nn.Sequential(
            nn.Linear(
                RELIABILITY_VECTOR_DIM,
                RELIABILITY_EMBED_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                RELIABILITY_EMBED_DIM
            ),
        )

        self.reliability_aware_correction_gate = nn.Sequential(
            nn.Linear(
                JOINT_CONTEXT_DIM
                +
                RELIABILITY_EMBED_DIM,
                RELIABILITY_GATE_HIDDEN_DIM,
            ),
            nn.GELU(),
            nn.Dropout(
                RELIABILITY_GATE_DROPOUT
            ),
            nn.Linear(
                RELIABILITY_GATE_HIDDEN_DIM,
                1,
            ),
            nn.Sigmoid(),
        )

        # Stage10-C frozen initialization.
        nn.init.xavier_uniform_(
            self.reliability_encoder[
                0
            ].weight
        )

        nn.init.zeros_(
            self.reliability_encoder[
                0
            ].bias
        )

        nn.init.xavier_uniform_(
            self.reliability_aware_correction_gate[
                0
            ].weight
        )

        nn.init.zeros_(
            self.reliability_aware_correction_gate[
                0
            ].bias
        )

        nn.init.zeros_(
            self.reliability_aware_correction_gate[
                3
            ].weight
        )

        nn.init.constant_(
            self.reliability_aware_correction_gate[
                3
            ].bias,
            RELIABILITY_GATE_INITIAL_BIAS,
        )


    @staticmethod
    def image_reliability_vector(
        baseline_logits,
    ):

        if (
            baseline_logits.ndim
            !=
            2
            or
            baseline_logits.shape[
                1
            ]
            !=
            V2_NUM_CLASSES
        ):

            raise ValueError(
                "baseline_logits must have shape [B,9]."
            )

        probabilities = torch.softmax(
            baseline_logits.detach(),
            dim=1,
        )

        confidence = probabilities.max(
            dim=1
        ).values

        entropy = -(
            probabilities
            *
            torch.log(
                probabilities.clamp_min(
                    ENTROPY_EPSILON
                )
            )
        ).sum(
            dim=1
        )

        normalized_entropy = (
            entropy
            /
            math.log(
                V2_NUM_CLASSES
            )
        )

        certainty = (
            1.0
            -
            normalized_entropy
        ).clamp(
            0.0,
            1.0,
        )

        top2 = torch.topk(
            probabilities,
            k=2,
            dim=1,
        ).values

        margin = (
            top2[
                :,
                0
            ]
            -
            top2[
                :,
                1
            ]
        ).clamp(
            0.0,
            1.0,
        )

        reliability = torch.stack(
            [
                confidence,
                certainty,
                margin,
            ],
            dim=1,
        )

        return reliability


    def forward(
        self,
        image,
        morphology,
        domain_context,
        return_aux=True,
    ):

        v1_outputs = super().forward(
            image=image,
            morphology=morphology,
            domain_context=domain_context,
            return_aux=True,
        )

        baseline_logits = v1_outputs[
            "baseline_logits"
        ]

        delta_logits = v1_outputs[
            "delta_logits"
        ]

        joint_context_embedding = v1_outputs[
            "joint_context_embedding"
        ]

        reliability_vector = (
            self.image_reliability_vector(
                baseline_logits
            )
        )

        reliability_embedding = (
            self.reliability_encoder(
                reliability_vector
            )
        )

        correction_gate = (
            self.reliability_aware_correction_gate(
                torch.cat(
                    [
                        joint_context_embedding,
                        reliability_embedding,
                    ],
                    dim=1,
                )
            )
        )

        final_logits = (
            baseline_logits
            +
            correction_gate
            *
            delta_logits
        )

        if not return_aux:

            return final_logits

        outputs = dict(
            v1_outputs
        )

        outputs[
            "v1_correction_gate_diagnostic_unused"
        ] = v1_outputs[
            "correction_gate"
        ]

        outputs[
            "final_logits"
        ] = final_logits

        outputs[
            "correction_gate"
        ] = correction_gate

        outputs[
            "reliability_vector"
        ] = reliability_vector

        outputs[
            "reliability_embedding"
        ] = reliability_embedding

        return outputs


    def inference(
        self,
        image,
        morphology,
        domain_context,
    ):

        return self.forward(
            image=image,
            morphology=morphology,
            domain_context=domain_context,
            return_aux=False,
        )


def compute_morphdg_v2_losses(
    outputs,
    class_targets,
    hard_domain_targets,
):

    # Exact authenticated V1 loss dictionary.
    losses = dict(
        compute_morphdg_losses(
            outputs,
            class_targets,
            hard_domain_targets,
        )
    )

    # Stage10-D-R1 authenticated aggregate key.
    if "total" not in losses:

        raise KeyError(
            "Authenticated V1 loss dictionary must contain 'total'."
        )

    baseline_probability = torch.softmax(
        outputs[
            "baseline_logits"
        ],
        dim=1,
    ).detach()

    final_log_probability = F.log_softmax(
        outputs[
            "final_logits"
        ],
        dim=1,
    )

    baseline_log_probability = torch.log(
        baseline_probability.clamp_min(
            ENTROPY_EPSILON
        )
    )

    per_sample_kl = (
        baseline_probability
        *
        (
            baseline_log_probability
            -
            final_log_probability
        )
    ).sum(
        dim=1
    )

    class_targets = class_targets.long()

    sample_weight = baseline_probability.gather(
        1,
        class_targets.view(
            -1,
            1,
        ),
    ).squeeze(
        1
    )

    class_evidence_preservation = (
        sample_weight
        *
        per_sample_kl
    ).mean()

    losses[
        "class_evidence_preservation"
    ] = class_evidence_preservation

    losses[
        "total"
    ] = (
        losses[
            "total"
        ]
        +
        CLASS_EVIDENCE_PRESERVATION_WEIGHT
        *
        class_evidence_preservation
    )

    ensure_finite_loss_dict(
        losses
    )

    return losses
