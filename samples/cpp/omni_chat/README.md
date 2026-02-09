# Omni Chat Sample

This sample demonstrates the OmniPipeline API for omni-modal models that can process and generate multiple modalities (text, images, audio, video).

## Features

The sample showcases:
- **Text-to-Text**: Standard text generation
- **Text-to-Speech**: Generate audio output from text
- **Audio Transcription**: Convert speech to text
- **Image Understanding**: Analyze and describe images
- **Multimodal Chat**: Interactive conversation with support for text, images, and audio

## Model Support

This sample is designed for omni-modal models like:
- GPT-4o (when exported to OpenVINO format)
- Qwen2-Audio
- Other models with similar multimodal architecture

## Prerequisites

1. Install OpenVINO GenAI:
```bash
pip install openvino-genai
```

2. Export a compatible model to OpenVINO format (example for a hypothetical omni model):
```bash
optimum-cli export openvino --model <model-name> --weight-format int4 model_dir
```

## Building

The sample is built with the standard OpenVINO GenAI build process:

```bash
cd <openvino.genai>/samples/cpp/omni_chat
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Usage

### Basic Usage

```bash
./omni_chat <MODEL_DIR> [IMAGE_FILE] [AUDIO_FILE] [DEVICE]
```

### Examples

1. **Text-only mode**:
```bash
./omni_chat ./gpt4o-model
```

2. **With image input**:
```bash
./omni_chat ./gpt4o-model image.jpg
```

3. **With image and audio inputs**:
```bash
./omni_chat ./gpt4o-model image.jpg speech.wav
```

4. **Use GPU**:
```bash
./omni_chat ./gpt4o-model image.jpg speech.wav GPU
```

## Interactive Chat Commands

When in chat mode, you can use these commands:

- `/voice <name>` - Set voice for audio output (e.g., alloy, echo, nova)
- `/audio` - Toggle audio output generation
- `/help` - Show available commands
- `exit` - Quit the chat

## API Highlights

### Initialize Pipeline
```cpp
ov::genai::OmniPipeline pipe("model_path", "CPU");
```

### Text Generation
```cpp
ov::genai::OmniGenerationConfig config;
config.max_new_tokens = 100;
auto result = pipe.generate("What is AI?", config, streamer);
```

### Text-to-Speech
```cpp
ov::genai::OmniGenerationConfig config;
config.output_modality = "audio";
config.voice = "alloy";
auto result = pipe.generate("Hello world", config);
// Audio available in result.audio
```

### Image Understanding
```cpp
ov::Tensor image = load_image("photo.jpg");
auto result = pipe.generate("Describe this image", image, config, streamer);
```

### Audio Transcription
```cpp
std::vector<float> audio_samples = load_audio("speech.wav");
auto result = pipe.generate("Transcribe", audio_samples, config, streamer);
```

### Multimodal Chat
```cpp
pipe.start_chat();
auto result1 = pipe.generate("Hi!", image, config, streamer);
auto result2 = pipe.generate("Tell me more", config, streamer);
pipe.finish_chat();
```

## Configuration Options

### OmniGenerationConfig Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_modality` | string | "text", "audio", or "text+audio" |
| `voice` | string | Voice preset for audio generation |
| `audio_sample_rate` | size_t | Sample rate in Hz (default: 24000) |
| `stream_audio` | bool | Enable real-time audio streaming |
| `max_audio_duration` | float | Max audio duration in seconds |
| `audio_temperature` | float | Temperature for audio generation |

Plus all standard GenerationConfig parameters (max_new_tokens, temperature, top_p, etc.)

## Performance Metrics

The OmniPerfMetrics provides detailed timing information:

```cpp
auto result = pipe.generate("prompt", config);
std::cout << "TTFT: " << result.omni_perf_metrics.get_ttft().mean << " ms\n";
std::cout << "TPOT: " << result.omni_perf_metrics.get_tpot().mean << " ms\n";
std::cout << "Audio encoding: " << result.omni_perf_metrics.audio_encoding_duration.mean << " ms\n";
```

## Notes

- Audio input should be mono, 16000 Hz (model-dependent)
- Images should be RGB uint8 tensors with [HWC] or [NHWC] layout
- Audio output format depends on model configuration (typically 24000 Hz)
- The sample includes placeholder audio/image loading functions - integrate with proper libraries like libsndfile and OpenCV for production use

## See Also

- [VLM Pipeline Sample](../visual_language_chat/) - Image+text processing
- [Whisper Sample](../whisper_speech_recognition/) - Audio transcription
- [Text Generation Sample](../text_generation/) - Text-only generation
