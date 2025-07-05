"""Debug Autoformer input requirements."""

import torch
import warnings
from transformers import AutoformerConfig, AutoformerForPrediction

warnings.filterwarnings("ignore")


def test_autoformer_minimal():
    """Test minimal Autoformer configuration."""
    print("Testing Autoformer with minimal config...")

    # Minimal config
    config = AutoformerConfig(
        context_length=32,
        prediction_length=1,
        num_time_features=0,
        lags_sequence=[1],
        num_dynamic_real_features=0,
        num_static_categorical_features=0,
        num_static_real_features=0,
        cardinality=[],
        embedding_dimension=[],
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
    )

    try:
        model = AutoformerForPrediction(config)
        print(f"✅ Model created successfully")

        # Test inputs
        batch_size = 2
        context_length = 32
        prediction_length = 1

        past_values = torch.randn(batch_size, context_length, 1)
        future_values = torch.randn(batch_size, prediction_length)
        past_time_features = torch.empty(batch_size, context_length, 0)
        past_observed_mask = torch.ones(batch_size, context_length, dtype=torch.bool)

        print(f"past_values: {past_values.shape}")
        print(f"future_values: {future_values.shape}")
        print(f"past_time_features: {past_time_features.shape}")
        print(f"past_observed_mask: {past_observed_mask.shape}")

        # Test forward pass
        outputs = model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_values=future_values,
        )

        print(f"✅ Forward pass successful")
        print(f"Output prediction shape: {outputs.prediction_outputs.shape}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_autoformer_minimal()
