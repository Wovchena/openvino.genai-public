# Omni Models API

This directory contains the API for Omni-modal models (like GPT-4o) that can process and generate multiple modalities including text, images, audio, and video.

## Overview

The Omni API provides a unified interface for models that support:
- **Input modalities**: Text, Images, Audio, Video
- **Output modalities**: Text, Audio, or both simultaneously
- **Conversational context**: Maintains chat history across modalities
- **Streaming**: Real-time text and audio streaming

## Architecture

The API consists of three main components:

### 1. OmniPipeline
The main pipeline class for loading and running omni-modal models.

```cpp
#include "openvino/genai/omni/pipeline.hpp"

ov::genai::OmniPipeline pipe("path/to/model", "CPU");
```

### 2. OmniGenerationConfig
Configuration for generation with omni-specific parameters.

```cpp
ov::genai::OmniGenerationConfig config;
config.max_new_tokens = 100;
config.output_modality = "text+audio";  // or "text" or "audio"
config.voice = "alloy";
config.audio_sample_rate = 24000;
```

### 3. OmniDecodedResults
Results structure containing generated text and/or audio with performance metrics.

```cpp
auto result = pipe.generate("Hello", config);
std::cout << result.texts[0] << std::endl;
if (result.audio.has_value()) {
    // Process audio samples
    auto& audio_samples = result.audio.value();
}
```

## Usage Examples

### Text-to-Text Generation
```cpp
ov::genai::OmniPipeline pipe("gpt-4o-model", "GPU");
auto result = pipe.generate("What is OpenVINO?");
std::cout << result.texts[0] << std::endl;
```

### Image Understanding
```cpp
ov::Tensor image = /* load image as uint8 RGB tensor */;
auto result = pipe.generate("Describe this image", image);
std::cout << result.texts[0] << std::endl;
```

### Audio Transcription
```cpp
std::vector<float> audio_samples = /* load audio at 16000 Hz */;
auto result = pipe.generate("Transcribe this audio", audio_samples);
std::cout << result.texts[0] << std::endl;
```

### Text-to-Speech
```cpp
ov::genai::OmniGenerationConfig config;
config.output_modality = "audio";
config.voice = "alloy";

auto result = pipe.generate("Hello, how are you?", config);
if (result.audio.has_value()) {
    auto& samples = result.audio.value();
    // Save or play audio samples
}
```

### Multimodal Conversation
```cpp
pipe.start_chat();

// Turn 1: Image + Text
ov::Tensor image = /* load image */;
auto result1 = pipe.generate("What's in this image?", image);
std::cout << result1.texts[0] << std::endl;

// Turn 2: Text only (maintains context)
auto result2 = pipe.generate("Can you describe it in more detail?");
std::cout << result2.texts[0] << std::endl;

// Turn 3: Audio + Text
std::vector<float> audio = /* load audio */;
auto result3 = pipe.generate("What did they say?", audio);
std::cout << result3.texts[0] << std::endl;

pipe.finish_chat();
```

### Multimodal Output
```cpp
ov::genai::OmniGenerationConfig config;
config.output_modality = "text+audio";
config.voice = "nova";

auto result = pipe.generate("Tell me a joke", config);
std::cout << "Text: " << result.texts[0] << std::endl;
std::cout << "Audio length: " << result.audio->size() / result.audio_sample_rate << "s" << std::endl;
```

## Python API

```python
import openvino_genai as ov_genai
import numpy as np

# Initialize pipeline
pipe = ov_genai.OmniPipeline("path/to/model", "CPU")

# Text-to-text
result = pipe.generate("What is AI?")
print(result.texts[0])

# Image understanding
image = np.array(...)  # RGB image as numpy array
result = pipe.generate("Describe this image", image=image)
print(result.texts[0])

# Audio transcription
audio_samples = np.array(...)  # Float audio samples
result = pipe.generate("Transcribe", audio=audio_samples)
print(result.texts[0])

# Text-to-speech
config = ov_genai.OmniGenerationConfig()
config.output_modality = "audio"
config.voice = "alloy"
result = pipe.generate("Hello!", config)
if result.audio is not None:
    # result.audio is a numpy array of float samples
    print(f"Generated {len(result.audio)} audio samples")

# Chat mode
pipe.start_chat()
result1 = pipe.generate("Hi there!", image=image)
result2 = pipe.generate("What else can you see?")
pipe.finish_chat()
```

## Configuration Options

### OmniGenerationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_modality` | string | "text" | Output type: "text", "audio", or "text+audio" |
| `voice` | string | "default" | Voice preset for audio generation (e.g., "alloy", "echo", "nova") |
| `audio_sample_rate` | size_t | 24000 | Sample rate in Hz for audio output |
| `stream_audio` | bool | false | Enable real-time audio streaming |
| `max_audio_duration` | float | 0.0 | Maximum audio duration in seconds (0 = no limit) |
| `audio_temperature` | float | 1.0 | Temperature for audio generation randomness |

All standard `GenerationConfig` parameters are also supported (max_new_tokens, temperature, top_p, etc.).

## Performance Metrics

`OmniPerfMetrics` extends `PerfMetrics` with omni-specific timing:
- `audio_encoding_duration`: Time spent encoding audio inputs
- `image_encoding_duration`: Time spent encoding image inputs
- `video_encoding_duration`: Time spent encoding video inputs
- `audio_decoding_duration`: Time spent generating audio outputs
- `fusion_duration`: Time spent fusing multimodal representations

```cpp
auto result = pipe.generate("prompt", config);
std::cout << "TTFT: " << result.omni_perf_metrics.get_ttft().mean << " ms" << std::endl;
std::cout << "Audio encoding: " << result.omni_perf_metrics.audio_encoding_duration.mean << " ms" << std::endl;
```

## Model Requirements

Omni models should provide the following components:
- **Text Tokenizer**: Standard HuggingFace tokenizer
- **Text Encoder**: Encodes text tokens to embeddings
- **Image Encoder**: Encodes images to embeddings (if image support needed)
- **Audio Encoder**: Encodes audio to embeddings (if audio input support needed)
- **Video Encoder**: Encodes video frames to embeddings (if video support needed)
- **Decoder**: Generates text output tokens
- **Audio Decoder**: Generates audio samples (if audio output support needed)
- **config.json**: Model configuration including modality settings

## Supported Models

Models compatible with this API:
- GPT-4o family (when exported to OpenVINO format)
- Qwen2-Audio
- Other omni-modal models following similar architecture

## Integration Notes

The Omni API follows OpenVINO GenAI design patterns:
- Consistent with `VLMPipeline` for visual inputs
- Compatible with `WhisperPipeline` audio format
- Extends `DecodedResults` and `PerfMetrics` base classes
- Supports property-based configuration like other pipelines
- Full device abstraction (CPU, GPU, NPU)
