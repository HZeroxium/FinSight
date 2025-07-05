"""Test TimeSeriesTransformer model inputs directly."""

import torch
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


def test_ts_transformer():
    # Simple config
    config = TimeSeriesTransformerConfig(
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

    model = TimeSeriesTransformerForPrediction(config)

    # Create simple inputs
    batch_size = 2
    context_length = 32
    prediction_length = 1

    past_values = torch.randn(batch_size, context_length, 1)
    future_values = torch.randn(batch_size, prediction_length)

    print(f"past_values shape: {past_values.shape}")
    print(f"future_values shape: {future_values.shape}")

    # Test different input combinations
    inputs_combinations = [
        # Just basic inputs
        {
            "past_values": past_values,
            "future_values": future_values,
        },
        # With observed mask
        {
            "past_values": past_values,
            "future_values": future_values,
            "past_observed_mask": torch.ones(
                batch_size, context_length, dtype=torch.bool
            ),
        },
        # With time features (empty)
        {
            "past_values": past_values,
            "future_values": future_values,
            "past_observed_mask": torch.ones(
                batch_size, context_length, dtype=torch.bool
            ),
            "past_time_features": torch.empty(batch_size, context_length, 0),
        },
    ]

    for i, inputs in enumerate(inputs_combinations):
        try:
            print(f"\nTest {i+1}: inputs keys = {list(inputs.keys())}")
            for key, value in inputs.items():
                if value is not None:
                    print(f"  {key}: {value.shape} {value.dtype}")
                else:
                    print(f"  {key}: None")

            output = model(**inputs)
            print(f"  SUCCESS: output shape = {output.last_hidden_state.shape}")
            break
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    test_ts_transformer()
