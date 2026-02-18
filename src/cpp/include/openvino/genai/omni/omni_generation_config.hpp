// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include "openvino/genai/generation_config.hpp"

namespace ov::genai {

/**
 * @brief Configuration for omni-modal generation.
 * Extends GenerationConfig with omni-specific parameters.
 */
struct OPENVINO_GENAI_EXPORTS OmniGenerationConfig : public GenerationConfig {
    /**
     * @brief Output modality types for generation.
     * Can be "text", "audio", or "text+audio" for multimodal output.
     */
    std::string output_modality = "text";

    /**
     * @brief Voice preset for audio output generation.
     * Used when output_modality includes "audio".
     */
    std::string voice = "default";

    /**
     * @brief Audio sample rate in Hz for audio output.
     * Default is 24000 Hz.
     */
    size_t audio_sample_rate = 24000;

    /**
     * @brief Enable audio streaming for real-time audio generation.
     */
    bool stream_audio = false;

    /**
     * @brief Maximum duration in seconds for generated audio.
     * 0 means no limit.
     */
    float max_audio_duration = 0.0f;

    /**
     * @brief Temperature for audio generation.
     * Controls randomness in audio output. Higher values make output more random.
     */
    float audio_temperature = 1.0f;

    OmniGenerationConfig() = default;
    explicit OmniGenerationConfig(const GenerationConfig& base) : GenerationConfig(base) {}
};

} // namespace ov::genai
