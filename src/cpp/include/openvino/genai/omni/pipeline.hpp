// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <optional>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/omni/omni_generation_config.hpp"
#include "openvino/genai/omni/perf_metrics.hpp"

namespace ov::genai {

/**
 * @brief Audio data representation.
 * Audio is represented as raw float samples.
 */
using RawAudioInput = std::vector<float>;

/**
 * @brief Results structure for omni-modal generation.
 * Contains generated text, audio, and performance metrics.
 */
class OPENVINO_GENAI_EXPORTS OmniDecodedResults : public DecodedResults {
public:
    /**
     * @brief Generated audio output as raw float samples.
     * Contains audio data when output_modality includes "audio".
     * Format: vector of float samples, mono or interleaved stereo.
     */
    std::optional<std::vector<float>> audio;

    /**
     * @brief Audio sample rate in Hz.
     * Valid when audio output is present.
     */
    size_t audio_sample_rate = 0;

    /**
     * @brief Number of audio channels (1 for mono, 2 for stereo).
     */
    size_t audio_channels = 1;

    /**
     * @brief Performance metrics specific to omni-modal generation.
     */
    OmniPerfMetrics omni_perf_metrics;
};

/**
 * @brief Omni-modal pipeline for models that can process and generate
 * multiple modalities (text, images, audio, video).
 * 
 * This pipeline supports models like GPT-4o that can:
 * - Accept text, image, audio, and video inputs
 * - Generate text and/or audio outputs
 * - Maintain conversational context across modalities
 * 
 * Example usage:
 * @code
 * OmniPipeline pipe("path/to/model", "CPU");
 * 
 * // Text + Image input -> Text output
 * auto result1 = pipe.generate("Describe this image", image_tensor);
 * 
 * // Text + Audio input -> Text output
 * auto result2 = pipe.generate("Transcribe this audio", audio_samples);
 * 
 * // Text input -> Audio output
 * OmniGenerationConfig config;
 * config.output_modality = "audio";
 * config.voice = "alloy";
 * auto result3 = pipe.generate("Hello, world!", config);
 * 
 * // Multimodal conversation
 * pipe.start_chat();
 * auto result4 = pipe.generate("What's in this image?", image_tensor);
 * auto result5 = pipe.generate("Can you describe it in more detail?");
 * pipe.finish_chat();
 * @endcode
 */
class OPENVINO_GENAI_EXPORTS OmniPipeline {
public:
    /**
     * @brief Construct a pipeline from a folder containing tokenizer and model IRs.
     * @param models_path A folder to read tokenizer and model IRs.
     * @param device Inference device. A tokenizer is always compiled for CPU.
     * @param properties A config to pass to ov::Core::compile_model().
     */
    OmniPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /**
     * @brief Construct a pipeline from a map of models and their weights.
     * @param models_map A map where key is model name (e.g., "text_encoder", "audio_encoder",
     *                   "image_encoder", "decoder", "audio_decoder") and value is a pair of
     *                   model IR as string and weights as tensor.
     * @param tokenizer A tokenizer.
     * @param config_dir_path A path to directory containing config.json.
     * @param device Inference device. A tokenizer is always compiled for CPU.
     * @param properties A config to pass to ov::Core::compile_model().
     * @param generation_config Optional generation configuration for the pipeline.
     */
    OmniPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties = {},
        const ov::genai::OmniGenerationConfig& generation_config = {}
    );

    /**
     * @brief Construct a pipeline from a folder containing tokenizer and model IRs.
     * Accepts arbitrary list of optional properties.
     * @param models_path A folder to read tokenizer and model IRs.
     * @param device Inference device. A tokenizer is always compiled for CPU.
     * @param properties A config to pass to ov::Core::compile_model().
     */
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    OmniPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        Properties&&... properties)
        : OmniPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /**
     * @brief Construct a pipeline from a map of models and their weights.
     * @param models_map A map where key is model name and value is a pair of
     *                   model IR as string and weights as tensor.
     * @param tokenizer A tokenizer.
     * @param config_dir_path A path to directory containing config.json.
     * @param device Inference device. A tokenizer is always compiled for CPU.
     * @param properties A config to pass to ov::Core::compile_model().
     */
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    OmniPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        Properties&&... properties)
        : OmniPipeline(models_map, tokenizer, config_dir_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /**
     * @brief Default destructor.
     */
    ~OmniPipeline();

    /**
     * @brief Generate a response given a text prompt.
     * @param prompt A prompt to respond to.
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const std::string& prompt,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a text prompt and images.
     * @param prompt A prompt to respond to.
     * @param images Images to be processed. Format: uint8 RGB with [NHWC] or [HWC] layout.
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a text prompt and a single image.
     * @param prompt A prompt to respond to.
     * @param image Image to be processed. Format: uint8 RGB with [NHWC] or [HWC] layout.
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const std::string& prompt,
        const ov::Tensor& image,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a text prompt and audio input.
     * @param prompt A prompt to respond to.
     * @param audio Audio samples as raw float values. Format: mono audio, sample rate 
     *              should match model's expected input sample rate (typically 16000 Hz).
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const std::string& prompt,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a text prompt, images, and audio.
     * @param prompt A prompt to respond to.
     * @param images Images to be processed. Format: uint8 RGB with [NHWC] or [HWC] layout.
     * @param audio Audio samples as raw float values.
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a text prompt, images, videos, and audio.
     * @param prompt A prompt to respond to.
     * @param images Images to be processed. Format: uint8 RGB with [NHWC] or [HWC] layout.
     * @param videos Videos to be processed. Each video is a tensor with multiple frames.
     *               Format: uint8 RGB with [NFHWC] or [FHWC] layout (F = frames).
     * @param audio Audio samples as raw float values.
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a prompt and config map.
     * @param prompt A prompt to respond to.
     * @param config_map A config may contain OmniGenerationConfig, values for its members,
     *                   StreamerVariant, images, videos, and/or audio.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    );

    /**
     * @brief Generate a response given a prompt and arbitrary number of ov::Property instances.
     * Example:
     * generate("text", image(rgb), audio(samples), output_modality("audio"));
     * @param prompt A prompt to respond to.
     * @param ...properties ov::Property instances to be combined into ov::AnyMap.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<OmniDecodedResults, Properties...> generate(
        const std::string& prompt,
        Properties&&... properties
    ) {
        return generate(
            prompt, AnyMap{std::forward<Properties>(properties)...}
        );
    }

    /**
     * @brief Generate a response given a chat history.
     * @param history Chat history with messages.
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const ChatHistory& history,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a chat history and images.
     * @param history Chat history with messages.
     * @param images Images to be associated with the last chat history user message.
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a chat history and audio.
     * @param history Chat history with messages.
     * @param audio Audio samples to be associated with the last chat history user message.
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const ChatHistory& history,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a chat history, images, videos, and audio.
     * @param history Chat history with messages.
     * @param images Images to be associated with the last chat history user message.
     * @param videos Videos to be associated with the last chat history user message.
     * @param audio Audio samples to be associated with the last chat history user message.
     * @param generation_config A config to follow for generation.
     * @param streamer A streamer to acquire intermediate result.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /**
     * @brief Generate a response given a chat history and config map.
     * @param history Chat history with messages.
     * @param config_map A config may contain OmniGenerationConfig, values for its members,
     *                   StreamerVariant, images, videos, and/or audio.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    OmniDecodedResults generate(
        const ChatHistory& history,
        const ov::AnyMap& config_map
    );

    /**
     * @brief Generate a response given a chat history and arbitrary number of ov::Property instances.
     * @param history Chat history with messages.
     * @param ...properties ov::Property instances to be combined into ov::AnyMap.
     * @return OmniDecodedResults structure containing generated text/audio and perf metrics.
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<OmniDecodedResults, Properties...> generate(
        const ChatHistory& history,
        Properties&&... properties
    ) {
        return generate(
            history, AnyMap{std::forward<Properties>(properties)...}
        );
    }

    /**
     * @brief Activate chat mode. Chat preserves previous history.
     * Calling start_chat() again or finish_chat() drops the memorized history.
     * @param system_message Some chat_templates contain system role in addition to
     *                       user and assistant roles. Set a message for that role.
     */
    void start_chat(const std::string& system_message = "");

    /**
     * @brief Deactivate chat mode.
     */
    void finish_chat();

    /**
     * @brief Set a custom chat template.
     * @param new_template A new template to override with.
     */
    void set_chat_template(const std::string& new_template);

    /**
     * @brief Get a Tokenizer used to tokenize input and detokenize output.
     * @return Tokenizer used by the pipeline.
     */
    ov::genai::Tokenizer get_tokenizer() const;

    /**
     * @brief Extract OmniGenerationConfig used to get default values.
     * @return Default values used.
     */
    OmniGenerationConfig get_generation_config() const;

    /**
     * @brief Override default values for OmniGenerationConfig.
     * @param new_config A config to override default values with.
     */
    void set_generation_config(const OmniGenerationConfig& new_config);

protected:
    // Base class for pipeline implementations (regular and continuous batching)
    class OmniPipelineBase {
    public:
        virtual ~OmniPipelineBase() = default;

        // Pure virtual methods that implementations must provide
        virtual OmniDecodedResults generate(
            const std::string& prompt,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual OmniDecodedResults generate(
            const std::string& prompt,
            const std::vector<ov::Tensor>& images,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual OmniDecodedResults generate(
            const std::string& prompt,
            const ov::Tensor& image,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual OmniDecodedResults generate(
            const std::string& prompt,
            const RawAudioInput& audio,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual OmniDecodedResults generate(
            const std::string& prompt,
            const std::vector<ov::Tensor>& images,
            const RawAudioInput& audio,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual OmniDecodedResults generate(
            const std::string& prompt,
            const std::vector<ov::Tensor>& images,
            const std::vector<ov::Tensor>& videos,
            const RawAudioInput& audio,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual OmniDecodedResults generate(
            const ChatHistory& history,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual OmniDecodedResults generate(
            const ChatHistory& history,
            const std::vector<ov::Tensor>& images,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual OmniDecodedResults generate(
            const ChatHistory& history,
            const RawAudioInput& audio,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual OmniDecodedResults generate(
            const ChatHistory& history,
            const std::vector<ov::Tensor>& images,
            const std::vector<ov::Tensor>& videos,
            const RawAudioInput& audio,
            const OmniGenerationConfig& generation_config,
            const StreamerVariant& streamer
        ) = 0;

        virtual void start_chat(const std::string& system_message) = 0;
        virtual void finish_chat() = 0;
        virtual void set_chat_template(const std::string& new_template) = 0;
        virtual Tokenizer get_tokenizer() const = 0;
        virtual OmniGenerationConfig get_generation_config() const = 0;
        virtual void set_generation_config(const OmniGenerationConfig& new_config) = 0;

        virtual void set_load_time(float load_time) { m_load_time = load_time; }
        virtual float get_load_time() const { return m_load_time; }

    protected:
        float m_load_time = 0.0f;
    };

    // Regular implementation (to be defined in implementation file)
    class OmniPipelineImpl;
    
    // Continuous batching adapter (forward declaration)
    class OmniContinuousBatchingAdapter;

private:
    std::unique_ptr<OmniPipelineBase> m_pimpl;
};

/**
 * Utils that allow to use generate() in the following way:
 * pipe.generate(prompt, ov::genai::audio(audio_samples));
 * pipe.generate(prompt, ov::genai::output_modality("audio"));
 */
static constexpr ov::Property<RawAudioInput> audio{"audio"};
static constexpr ov::Property<std::string> output_modality{"output_modality"};
static constexpr ov::Property<std::string> voice{"voice"};
static constexpr ov::Property<size_t> audio_sample_rate{"audio_sample_rate"};
static constexpr ov::Property<bool> stream_audio{"stream_audio"};

} // namespace ov::genai
