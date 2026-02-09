// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

namespace ov::genai {

/**
 * @brief Adapter that enables ContinuousBatching mode for OmniPipeline.
 * 
 * This adapter wraps ContinuousBatchingPipeline to provide efficient batch processing
 * for omni-modal models. It's automatically selected when:
 * - SchedulerConfig is explicitly provided
 * - Paged Attention backend is detected and model supports it
 * - Device is not NPU (NPU always uses regular mode)
 * 
 * Benefits of ContinuousBatching mode:
 * - Higher throughput for multiple concurrent requests
 * - Efficient KV-cache management via paging
 * - Dynamic batching with request queueing
 * - Better resource utilization
 */
class OmniContinuousBatchingAdapter : public OmniPipeline::OmniPipelineImpl {
public:
    /**
     * @brief Construct adapter from models directory.
     * @param models_dir Path to directory containing model files.
     * @param scheduler_config Configuration for continuous batching scheduler.
     * @param device Target device (CPU, GPU, etc.).
     * @param properties Additional pipeline properties.
     */
    OmniContinuousBatchingAdapter(
        const std::filesystem::path& models_dir,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties
    );

    /**
     * @brief Construct adapter from models map.
     * @param models_map Map of model components (text/audio/image encoders, decoders).
     * @param tokenizer Pre-initialized tokenizer.
     * @param config_dir_path Path to configuration directory.
     * @param scheduler_config Configuration for continuous batching scheduler.
     * @param device Target device (CPU, GPU, etc.).
     * @param properties Additional pipeline properties.
     * @param generation_config Default generation configuration.
     */
    OmniContinuousBatchingAdapter(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties,
        const OmniGenerationConfig& generation_config
    );

    // Override generate methods to use ContinuousBatchingPipeline
    OmniDecodedResults generate(
        const std::string& prompt,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    OmniDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    OmniDecodedResults generate(
        const std::string& prompt,
        const ov::Tensor& image,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    OmniDecodedResults generate(
        const std::string& prompt,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    OmniDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    OmniDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    OmniDecodedResults generate(
        const ChatHistory& history,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    OmniDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    OmniDecodedResults generate(
        const ChatHistory& history,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    OmniDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const RawAudioInput& audio,
        const OmniGenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    // Chat mode management
    void start_chat(const std::string& system_message) override;
    void finish_chat() override;

    // Configuration management
    Tokenizer get_tokenizer() const override;
    OmniGenerationConfig get_generation_config() const override;
    void set_generation_config(const OmniGenerationConfig& new_config) override;

    // Note: set_chat_template not supported in continuous batching mode
    void set_chat_template(const std::string& new_template) override {
        OPENVINO_THROW("set_chat_template is not supported in ContinuousBatching mode.");
    }

private:
    ContinuousBatchingPipeline m_impl;

    // Helper to convert OmniDecodedResults from base DecodedResults
    OmniDecodedResults convert_to_omni_results(
        const DecodedResults& base_results,
        const std::chrono::steady_clock::time_point& start_time
    );
};

} // namespace ov::genai
