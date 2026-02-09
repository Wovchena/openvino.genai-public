# API Design Summary: Omni Models

## Overview

This document summarizes the proposed API design for Omni-modal models in OpenVINO GenAI. Omni models (like GPT-4o, Qwen2-Audio) can process and generate multiple modalities including text, images, audio, and video.

## Design Goals

1. **Consistency**: Follow existing OpenVINO GenAI API patterns (VLMPipeline, LLMPipeline, WhisperPipeline)
2. **Flexibility**: Support various input/output modality combinations
3. **Ease of Use**: Intuitive API for common use cases
4. **Performance**: Built-in metrics for optimization
5. **Extensibility**: Easy to add new modalities or capabilities

## API Components

### 1. OmniPipeline (pipeline.hpp)

Main pipeline class for loading and running omni-modal models.

**Key Features:**
- Multiple constructors (from filesystem, from models map)
- Overloaded `generate()` methods for different input combinations
- Chat mode support (`start_chat()`, `finish_chat()`)
- Configuration management
- Property-based API for flexible usage

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
std::vector<float> audio = load_audio("speech.wav");
auto result = pipe.generate("Transcribe", audio);
```

### 4. Text-to-Speech
```cpp
ov::genai::OmniGenerationConfig config;
config.output_modality = "audio";
config.voice = "alloy";
auto result = pipe.generate("Hello world", config);
// Use result.audio.value()
```

### 5. Multimodal Conversation
```cpp
pipe.start_chat();
auto r1 = pipe.generate("What's this?", image);
auto r2 = pipe.generate("Tell me more");
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

### Modality Handling

1. **Audio**: `std::vector<float>` for samples (consistent with Whisper)
2. **Images**: `ov::Tensor` (consistent with VLMPipeline)
3. **Videos**: `std::vector<ov::Tensor>` (consistent with VLMPipeline)
4. **Text**: `std::string` (consistent with LLMPipeline)

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

## Files Structure

```
src/cpp/include/openvino/genai/omni/
├── pipeline.hpp                   # Main OmniPipeline API
├── omni_generation_config.hpp    # Configuration
├── perf_metrics.hpp               # Performance metrics
├── py_bindings_example.cpp       # Python bindings structure
└── README.md                      # Comprehensive documentation

samples/cpp/omni_chat/
├── omni_chat.cpp                 # C++ sample
├── CMakeLists.txt                # Build configuration
└── README.md                     # Sample documentation
```

## Next Steps

For implementation:

1. Implement C++ core logic following VLMPipeline patterns
2. Add Python bindings following existing binding structure
3. Create tests for each modality combination
4. Add model conversion utilities
5. Write comprehensive documentation
6. Optimize for performance
7. Add continuous batching support

## Conclusion

The proposed Omni API provides a clean, consistent, and powerful interface for omni-modal models in OpenVINO GenAI. It follows established patterns while introducing necessary new capabilities for handling multiple input and output modalities in a unified way.
