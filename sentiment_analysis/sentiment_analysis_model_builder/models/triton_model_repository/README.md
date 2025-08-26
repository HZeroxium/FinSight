# FinBERT Sentiment Analysis Model for Triton Inference Server

This directory contains the exported FinBERT sentiment analysis model optimized for deployment with NVIDIA Triton Inference Server.

## Model Overview

- **Model**: ProsusAI/finbert fine-tuned for financial sentiment analysis
- **Task**: 3-class sentiment classification (NEGATIVE, NEUTRAL, POSITIVE)
- **Format**: ONNX
- **Input**: Tokenized text (input_ids, attention_mask)
- **Output**: Classification logits

## Directory Structure

```
triton_model_repository/
├── finbert_sentiment/           # Model repository
│   ├── config.pbtxt            # Triton model configuration
│   └── 1/                      # Model version 1
│       └── model.onnx          # ONNX model file
├── triton_client_example.py    # Python client example
└── README.md                   # This file
```

## Model Configuration

The model is configured for:

- **Maximum batch size**: 64
- **Input sequence length**: 512 tokens
- **Dynamic batching**: Enabled with preferred batch sizes [4, 8, 16]
- **GPU acceleration**: TensorRT optimization with FP16 precision
- **Platform**: onnxruntime_onnx

## Input/Output Specification

### Inputs

1. **input_ids**: `[batch_size, 512]` - Tokenized input text (INT64)
2. **attention_mask**: `[batch_size, 512]` - Attention mask (INT64)

### Outputs

1. **logits**: `[batch_size, 3]` - Classification logits (FLOAT32)
   - Index 0: NEGATIVE
   - Index 1: NEUTRAL
   - Index 2: POSITIVE

## Deployment

### 1. Start Triton Inference Server

```bash
# Using Docker
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /path/to/triton_model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### 2. Verify Model Loading

```bash
# Check server health
curl -v localhost:8000/v2/health/ready

# Check model status
curl -v localhost:8000/v2/models/finbert_sentiment
```

### 3. Run Inference

Use the provided Python client:

```python
from triton_client_example import TritonFinBERTClient

client = TritonFinBERTClient()
predictions = client.predict([
    "Bitcoin is reaching new all-time highs!",
    "The market is highly volatile today."
])

for pred in predictions:
    print(f"Text: {pred['text']}")
    print(f"Sentiment: {pred['predicted_label']} ({pred['confidence']:.4f})")
```

## Performance Optimization

The model configuration includes several optimizations:

1. **Dynamic Batching**: Automatically batches requests for better throughput
2. **TensorRT**: GPU acceleration with FP16 precision
3. **Memory Management**: Optimized workspace size (1GB)
4. **Queue Management**: Low latency queue delay (100μs)

## API Endpoints

- **Health Check**: `GET /v2/health/ready`
- **Model Info**: `GET /v2/models/finbert_sentiment`
- **Inference**: `POST /v2/models/finbert_sentiment/infer`

## Dependencies

For the Python client:

```bash
pip install tritonclient[http] transformers torch numpy
```

## Example Usage

```python
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

# Initialize
client = httpclient.InferenceServerClient(url="localhost:8000")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# Prepare input
text = "The cryptocurrency market is bullish today"
encoding = tokenizer(text, max_length=512, padding="max_length",
                    truncation=True, return_tensors="np")

# Create inputs
inputs = [
    httpclient.InferInput("input_ids", encoding["input_ids"].shape, "INT64"),
    httpclient.InferInput("attention_mask", encoding["attention_mask"].shape, "INT64")
]
inputs[0].set_data_from_numpy(encoding["input_ids"].astype(np.int64))
inputs[1].set_data_from_numpy(encoding["attention_mask"].astype(np.int64))

# Run inference
outputs = [httpclient.InferRequestedOutput("logits")]
response = client.infer("finbert_sentiment", inputs, outputs=outputs)

# Get results
logits = response.as_numpy("logits")
predicted_class = np.argmax(logits[0])
labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
print(f"Predicted sentiment: {labels[predicted_class]}")
```

## Monitoring

Monitor the model performance using:

- Triton metrics endpoint: `GET /metrics`
- Model statistics: `GET /v2/models/finbert_sentiment/stats`

## Troubleshooting

### Common Issues

1. **Model Not Loading**

   - Check model path and permissions
   - Verify ONNX model integrity
   - Check Triton logs for errors

2. **Performance Issues**

   - Adjust batch sizes in config.pbtxt
   - Monitor GPU memory usage
   - Consider TensorRT optimization

3. **Input/Output Errors**
   - Verify input shapes match config
   - Check data types (INT64 for tokens, FLOAT32 for logits)
   - Ensure proper tokenization

### Logs and Debugging

```bash
# View Triton logs
docker logs <triton_container_id>

# Enable verbose logging
tritonserver --model-repository=/models --log-verbose=1
```

## Model Information

- **Training Data**: Financial news articles with sentiment labels
- **Validation Accuracy**: ~37.5% (on test set)
- **Model Size**: ~418 MB
- **Inference Latency**: ~10-50ms (depending on batch size and hardware)

For more details about the model training and evaluation, see the training logs and MLflow experiments.
