"""Phase 6 method implementations."""

from llm4rec.methods.calm_rec import (
    CALMConfig,
    CALMRecRanker,
    IntentEncoder,
    IntentSet,
    ItemEncoder,
    MockIntentEncoder,
    MockItemEncoder,
    RhoCoeffs,
    reliability_auc,
)
from llm4rec.methods.calm_encoders import (
    HashedItemEncoder,
    LexiconIntentEncoder,
    QwenIntentEncoder,
    QwenItemEncoder,
    build_encoders,
)
from llm4rec.methods.calm_trainer import (
    CALMLossSpec,
    CALMScaffoldTrainer,
    CALMTrainPlan,
    build_train_only_stats,
    calibrate_rho_on_validation,
    stage_2p5_reliability_gate,
)
from llm4rec.methods.calm_weak_labels import AttributeLexicon, default_beauty_lexicon
from llm4rec.methods.ours_method import OursMethodRanker
from llm4rec.methods.scalr import (
    MockPanelScorer,
    MonotoneCalibration,
    PanelScorer,
    PanelScoreRequest,
    SCALRConfig,
    SCALRRanker,
)
from llm4rec.methods.uncertainty_policy import PolicyDecision, UncertaintyPolicy

__all__ = [
    "OursMethodRanker",
    "PolicyDecision",
    "UncertaintyPolicy",
    "SCALRRanker",
    "SCALRConfig",
    "PanelScorer",
    "PanelScoreRequest",
    "MonotoneCalibration",
    "MockPanelScorer",
    "CALMRecRanker",
    "CALMConfig",
    "RhoCoeffs",
    "IntentSet",
    "ItemEncoder",
    "IntentEncoder",
    "MockItemEncoder",
    "MockIntentEncoder",
    "reliability_auc",
    "HashedItemEncoder",
    "LexiconIntentEncoder",
    "QwenItemEncoder",
    "QwenIntentEncoder",
    "build_encoders",
    "AttributeLexicon",
    "default_beauty_lexicon",
    "CALMLossSpec",
    "CALMScaffoldTrainer",
    "CALMTrainPlan",
    "build_train_only_stats",
    "calibrate_rho_on_validation",
    "stage_2p5_reliability_gate",
]


