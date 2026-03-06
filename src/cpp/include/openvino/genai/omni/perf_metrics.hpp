// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/perf_metrics.hpp"

namespace ov::genai {

/**
 * @brief Raw performance metrics specific to omni-modal models.
 */
struct OmniRawPerfMetrics {
    /** @brief Duration for each audio encoding call */
    std::vector<MicroSeconds> audio_encoding_durations;
    
    /** @brief Duration for each image encoding call */
    std::vector<MicroSeconds> image_encoding_durations;
    
    /** @brief Duration for each video encoding call */
    std::vector<MicroSeconds> video_encoding_durations;
    
    /** @brief Duration for each audio decoding call */
    std::vector<MicroSeconds> audio_decoding_durations;
    
    /** @brief Duration for multimodal fusion processing */
    std::vector<MicroSeconds> fusion_durations;
};

/**
 * @brief Performance metrics for omni-modal generation.
 * Extends PerfMetrics with omni-specific timing information.
 */
struct OPENVINO_GENAI_EXPORTS OmniPerfMetrics : public PerfMetrics {
    /** @brief Mean and standard deviation of audio encoding duration in milliseconds */
    MeanStdPair audio_encoding_duration;
    
    /** @brief Mean and standard deviation of image encoding duration in milliseconds */
    MeanStdPair image_encoding_duration;
    
    /** @brief Mean and standard deviation of video encoding duration in milliseconds */
    MeanStdPair video_encoding_duration;
    
    /** @brief Mean and standard deviation of audio decoding duration in milliseconds */
    MeanStdPair audio_decoding_duration;
    
    /** @brief Mean and standard deviation of multimodal fusion duration in milliseconds */
    MeanStdPair fusion_duration;

    OmniPerfMetrics() = default;
    OmniPerfMetrics(const PerfMetrics& perf_metrics) : PerfMetrics(perf_metrics) {}

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;

    OmniPerfMetrics operator+(const OmniPerfMetrics& metrics) const;
    OmniPerfMetrics& operator+=(const OmniPerfMetrics& right);

    OmniRawPerfMetrics omni_raw_metrics;
};

} // namespace ov::genai
