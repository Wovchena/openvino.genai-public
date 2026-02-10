# API Design Summary: Omni Models

## Overview

This document summarizes the proposed API design for Omni-modal models in OpenVINO GenAI. Omni models (like GPT-4o, Qwen2-Audio) can process and generate multiple modalities including text, images, audio, and video.

The API supports two execution modes:
- **Regular mode**: Optimized for single-request, lowest latency scenarios
- **ContinuousBatching mode**: Optimized for high-throughput, multiple concurrent requests

> **Design Decision**: The separate adapter pattern is used for ContinuousBatching integration, following the established patterns in VLMPipeline and LLMPipeline. See [DESIGN_DECISION_CONTINUOUS_BATCHING.md](DESIGN_DECISION_CONTINUOUS_BATCHING.md) for detailed rationale.

## Design Goals

1. **Consistency**: Follow existing OpenVINO GenAI API patterns (VLMPipeline, LLMPipeline, WhisperPipeline)
2. **Flexibility**: Support various input/output modality combinations
3. **Ease of Use**: Intuitive API for common use cases
4. **Performance**: Built-in metrics for optimization
5. **Extensibility**: Easy to add new modalities or capabilities
6. **Maintainability**: Clear separation of concerns between execution modes

## API Components

### 1. OmniPipeline (pipeline.hpp)

Main pipeline class for loading and running omni-modal models.

**Execution Modes:**
- **Regular mode**: Single request processing, optimized for lowest latency
- **ContinuousBatching mode**: Multiple concurrent requests, optimized for throughput

The pipeline automatically selects the appropriate mode based on:
- Explicit SchedulerConfig in properties → ContinuousBatching
- Paged Attention backend detection (if model supports it) → ContinuousBatching
- NPU device → Always regular mode
- Otherwise → Regular mode

**Key Features:**
- Multiple constructors (from filesystem, from models map)
- Overloaded `generate()` methods for different input combinations
- Chat mode support (`start_chat()`, `finish_chat()`)
- Configuration management
- Property-based API for flexible usage
- Dual-mode architecture (OmniPipelineImpl and OmniContinuousBatchingAdapter)

**Supported Input Combinations:**
- Text only
- Text + Image(s)
- Text + Audio
- Text + Video(s)
- Text + Image(s) + Audio
- Text + Image(s) + Video(s) + Audio
- ChatHistory + any of the above

**Example:**
```cpp
ov::genai::OmniPipeline pipe("model_path", "GPU");
auto result = pipe.generate("What's in this image?", image_tensor, config);
```

### 2. OmniGenerationConfig (omni_generation_config.hpp)

Configuration extending GenerationConfig with omni-specific parameters.

**New Parameters:**
- `output_modality`: "text", "audio", or "text+audio"
- `voice`: Voice preset for audio generation
- `audio_sample_rate`: Sample rate for audio output
- `stream_audio`: Enable real-time audio streaming
- `max_audio_duration`: Maximum audio duration limit
- `audio_temperature`: Temperature for audio generation

**Example:**
```cpp
ov::genai::OmniGenerationConfig config;
config.max_new_tokens = 100;
config.output_modality = "text+audio";
config.voice = "alloy";
```

### 3. OmniPerfMetrics (perf_metrics.hpp)

Performance metrics extending PerfMetrics with modality-specific timing.

**New Metrics:**
- `audio_encoding_duration`: Audio input encoding time
- `image_encoding_duration`: Image input encoding time
- `video_encoding_duration`: Video input encoding time
- `audio_decoding_duration`: Audio output generation time
- `fusion_duration`: Multimodal fusion processing time

**Example:**
```cpp
auto result = pipe.generate("prompt", config);
std::cout << "TTFT: " << result.omni_perf_metrics.get_ttft().mean << " ms\n";
std::cout << "Audio encoding: " << result.omni_perf_metrics.audio_encoding_duration.mean << " ms\n";
```

### 4. OmniDecodedResults (pipeline.hpp)

Results structure containing generated outputs and metrics.

**Fields:**
- Inherits: `texts`, `scores`, `perf_metrics` from DecodedResults
- `audio`: Optional vector of float audio samples
- `audio_sample_rate`: Sample rate of generated audio
- `audio_channels`: Number of audio channels (1=mono, 2=stereo)
- `omni_perf_metrics`: Omni-specific performance metrics

### 5. OmniContinuousBatchingAdapter (continuous_batching_adapter.hpp)

Adapter class that enables ContinuousBatching mode for OmniPipeline.

**Purpose:**
- Wraps ContinuousBatchingPipeline to provide high-throughput batch processing
- Automatically selected based on configuration and device capabilities
- Implements same interface as regular OmniPipelineImpl
- **Uses native audio API** from ContinuousBatchingPipeline (no conversion needed)

**Benefits:**
- **Higher throughput**: Process multiple requests concurrently
- **Efficient KV-cache**: Paged attention with dynamic memory management
- **Request queueing**: Automatic batching and scheduling
- **Better utilization**: Maximize GPU/CPU resource usage
- **Native audio support**: Leverages extended ContinuousBatchingPipeline API

**Usage:**
```cpp
// Explicit ContinuousBatching mode
ov::genai::SchedulerConfig scheduler_config;
scheduler_config.max_num_batched_tokens = 512;
scheduler_config.cache_size = 8;

ov::AnyMap properties;
properties.insert(ov::genai::scheduler_config(scheduler_config));

ov::genai::OmniPipeline pipe("model", "GPU", properties);
// Now using OmniContinuousBatchingAdapter internally
```

### 6. ContinuousBatchingPipeline Native Audio Extensions

**New in continuous_batching_pipeline.hpp:**

The base ContinuousBatchingPipeline API has been extended to natively support audio:

**Type Definitions:**
```cpp
using RawAudioInput = std::vector<float>;

class OmniDecodedResults : public VLMDecodedResults {
    std::optional<std::vector<float>> audio;
    size_t audio_sample_rate;
    size_t audio_channels;
};
```

**New Methods (9 overloads):**
- 3 audio-aware `add_request()` methods
- 6 audio-aware `generate()` methods returning `OmniDecodedResults`

**Benefits:**
- ✅ **Native support**: Audio is first-class citizen, not adapter workaround
- ✅ **Performance**: No conversion overhead
- ✅ **Consistency**: All modalities (text, images, videos, audio) handled uniformly
- ✅ **Reusability**: Other pipelines can leverage audio support

## Use Cases

### 1. Text-to-Text (Standard LLM)
```cpp
ov::genai::OmniPipeline pipe("model", "CPU");
auto result = pipe.generate("What is AI?");
std::cout << result.texts[0] << std::endl;
```

### 2. Image Understanding
```cpp
ov::Tensor image = load_image("photo.jpg");
auto result = pipe.generate("Describe this image", image);
```

### 3. Audio Transcription
```cpp
// Create audio tensor (1 second at 16kHz, mono)
ov::Tensor audio({16000}, ov::element::f32);
float* data = audio.data<float>();
// Fill with normalized audio samples in [-1.0, 1.0]
auto result = pipe.generate("Transcribe", audio);
```

### 4. Text-to-Speech
```cpp
ov::genai::OmniGenerationConfig config;
config.output_modality = "audio";
config.voice = "alloy";
auto result = pipe.generate("Hello world", config);
// result.audio.value() is ov::Tensor with shape [num_samples]
```

### 5. Multimodal Conversation
```cpp
pipe.start_chat();
auto r1 = pipe.generate("What's this?", image);
auto r2 = pipe.generate("Tell me more");
// Create audio tensor for input
ov::Tensor audio({16000}, ov::element::f32);
// ... fill audio ...
auto r3 = pipe.generate("Explain", audio);
pipe.finish_chat();
```

### 6. Simultaneous Text and Audio Output
```cpp
ov::genai::OmniGenerationConfig config;
config.output_modality = "text+audio";
auto result = pipe.generate("Tell me a story", config);
// Use both result.texts[0] and result.audio.value()
```

### 7. High-Throughput Batch Processing
```cpp
// Enable ContinuousBatching for multiple concurrent requests
ov::genai::SchedulerConfig scheduler_config;
scheduler_config.max_num_batched_tokens = 512;
scheduler_config.cache_size = 8;

ov::AnyMap properties;
properties.insert(ov::genai::scheduler_config(scheduler_config));

ov::genai::OmniPipeline pipe("model", "GPU", properties);

// Process multiple requests - automatically batched and scheduled
std::vector<std::string> prompts = {
    "Summarize this document",
    "Translate to Spanish",
    "What is quantum computing?"
};

for (const auto& prompt : prompts) {
    auto result = pipe.generate(prompt);
    std::cout << result.texts[0] << "\n\n";
}
```

## Python Bindings

Python API mirrors C++ with Pythonic conventions:

```python
import openvino_genai as ov_genai
import numpy as np

pipe = ov_genai.OmniPipeline("model_path", "CPU")

# Text generation
result = pipe.generate("Hello")

# With image
result = pipe.generate("Describe", image=np.array(...))

# With audio
result = pipe.generate("Transcribe", audio=np.array(...))

# Text-to-speech
config = ov_genai.OmniGenerationConfig()
config.output_modality = "audio"
result = pipe.generate("Hello", config)
```

## Design Rationale

### Following Existing Patterns

1. **Pipeline Architecture**: Like VLMPipeline and LLMPipeline
2. **Results Classes**: Extends DecodedResults pattern
3. **Configuration**: Extends GenerationConfig
4. **Metrics**: Extends PerfMetrics
5. **Property-based API**: Uses ov::Property for flexible configuration
6. **Tensor-based Modalities**: All modalities (images, videos, audio) use ov::Tensor

### Modality Handling

**Consistent Tensor Usage:**
1. **Images**: `ov::Tensor` with shape `[batch, height, width, channels]`
2. **Videos**: `ov::Tensor` with shape `[batch, frames, height, width, channels]`
3. **Audio**: `ov::Tensor` with shape `[num_samples]` or `[channels, num_samples]`
4. **Text**: `std::string` (tokenization handled internally)

**Rationale for ov::Tensor for Audio:**
- **API Consistency**: All visual/audio modalities use same type system
- **Shape Metadata**: Audio tensor shape is self-documenting
- **Type Safety**: Element type validation (f32 for normalized samples)
- **Memory Management**: Unified allocation and GPU readiness
- **Future-proof**: Enables batch audio, spatial audio, multi-channel processing
- **OpenVINO Native**: Seamless integration with model I/O tensors

**Audio Tensor Convention:**
- **Element type**: `ov::element::f32` (normalized float samples in `[-1.0, 1.0]`)
- **Mono audio**: Shape `[num_samples]` - 1D tensor
- **Stereo audio**: Shape `[2, num_samples]` - 2D tensor, channels first
- **Multi-channel**: Shape `[channels, num_samples]` - 2D tensor, channels first

### Output Flexibility

Supports three output modes via `output_modality`:
- `"text"`: Standard text generation
- `"audio"`: Audio-only output (TTS)
- `"text+audio"`: Simultaneous text and audio

### Chat Mode

Maintains conversational context across:
- Multiple turns
- Multiple modalities
- Mixed input types

## Implementation Considerations

### Model Requirements

An omni model should provide:
- Text tokenizer
- Encoders: text, image, audio, video (as needed)
- Decoders: text, audio (as needed)
- config.json with modality information

### Performance Optimization

- Lazy loading of encoders based on usage
- Efficient tensor management
- Streaming support for real-time output
- Model caching for repeated loads

### Memory Management

- Pass-by-reference for large tensors
- Move semantics for results
- Optional fields for unused modalities
- PIMPL pattern for implementation hiding

## Extension Points

The API is designed for future extensions:

1. **New Modalities**: Add new input/output types
2. **Streaming**: Enhanced streaming for audio chunks
3. **Batch Processing**: Multiple inputs simultaneously
4. **LoRA Support**: Adapter loading like LLMPipeline
5. **Advanced Features**: Cross-modal attention visualization, etc.

## Comparison with Existing Pipelines

| Feature | LLMPipeline | VLMPipeline | WhisperPipeline | OmniPipeline |
|---------|-------------|-------------|-----------------|--------------|
| Text Input | ✓ | ✓ | ✗ | ✓ |
| Image Input | ✗ | ✓ | ✗ | ✓ |
| Audio Input | ✗ | ✗ | ✓ | ✓ |
| Video Input | ✗ | ✓ | ✗ | ✓ |
| Text Output | ✓ | ✓ | ✓ | ✓ |
| Audio Output | ✗ | ✗ | ✗ | ✓ |
| Chat Mode | ✓ | ✓ | ✗ | ✓ |
| Streaming | ✓ | ✓ | ✗ | ✓ |
| ContinuousBatching | ✓ | ✓ | ✗ | ✓ |

## Files Structure

```
src/cpp/include/openvino/genai/omni/
├── pipeline.hpp                              # Main OmniPipeline API with dual-mode support
├── omni_generation_config.hpp               # Configuration
├── perf_metrics.hpp                          # Performance metrics
├── continuous_batching_adapter.hpp          # ContinuousBatching adapter implementation
├── py_bindings_example.cpp                  # Python bindings structure
├── README.md                                 # Comprehensive documentation
├── API_DESIGN_SUMMARY.md                    # Complete design rationale
└── DESIGN_DECISION_CONTINUOUS_BATCHING.md  # Detailed CB integration design decision

samples/cpp/omni_chat/
├── omni_chat.cpp                    # C++ sample
├── CMakeLists.txt                   # Build configuration
└── README.md                        # Sample documentation
```

## Next Steps

For implementation:

1. Implement C++ core logic following VLMPipeline patterns
2. Implement OmniContinuousBatchingAdapter for high-throughput mode
3. Add decision logic for automatic mode selection (regular vs continuous batching)
4. Add Python bindings following existing binding structure
5. Create tests for each modality combination and both execution modes
6. Add model conversion utilities
7. Write comprehensive documentation
8. Optimize for performance
9. Add benchmarks comparing regular and continuous batching modes

## Conclusion

The proposed Omni API provides a clean, consistent, and powerful interface for omni-modal models in OpenVINO GenAI. It follows established patterns while introducing necessary new capabilities for handling multiple input and output modalities in a unified way. The dual-mode architecture (regular and ContinuousBatching) ensures optimal performance for both low-latency single-request and high-throughput batch processing scenarios.
