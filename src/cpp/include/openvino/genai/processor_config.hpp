// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include <openvino/runtime/compiled_model.hpp>
#include <array>
#include <filesystem>

namespace ov::genai {
/// @brief A Configuration class passed to VLMPipeline and used to
/// change VLMPipeline's behavior.
class OPENVINO_GENAI_EXPORTS ProcessorConfig {
public:
    /// @brief Dimensions of the smaller, non-overlapping patches that the
    /// input image is divided into before being fed into the
    /// transformer model. Used to divide image height and width.
    size_t patch_size = 14;
    /// @brief A recommended size to resize an input image.
    size_t scale_resolution = 448;
    /// @brief Maximum allowed number of intput image slices.
    /// 0 disables slicing.
    size_t max_slice_nums = 0;
    /// @brief RGB values to be subtracted from image pixel values.
    /// Applied before norm_std.
    std::array<float, 3> norm_mean{0.0f, 0.0f, 0.0f};
    /// @brief RGB values to divide image pixel values.
    /// Applied after norm_mean.
    std::array<float, 3> norm_std{1.0f, 1.0f, 1.0f};
    /// @brief Default constructor
    ProcessorConfig() = default;
    /// @brief Construct ProcessorConfig from values in json_path.
    /// Keys in the file must match the ProcessorConfig's members.
    /// @param json_path A path to a file to extract the values from.
    explicit ProcessorConfig(const std::filesystem::path& json_path);
    /// @brief Default copy constructor.
    /// @param A config to copy from.
    ProcessorConfig(const ProcessorConfig&) = default;
};

/*
 * Utils that allow to use encode(), generate() and operator()() in the following way:
 * pipe.generate(input_ids, ov::genai::scale_resolution(448), ...)
 * pipe(input_ids, ov::genai::scale_resolution(448), ...)
*/
static constexpr ov::Property<size_t> patch_size{"patch_size"};
static constexpr ov::Property<size_t> scale_resolution{"scale_resolution"};
static constexpr ov::Property<size_t> max_slice_nums{"max_slice_nums"};
static constexpr ov::Property<std::array<float, 3>> norm_mean{"norm_mean"};
static constexpr ov::Property<std::array<float, 3>> norm_std{"norm_std"};
}  // namespace ov::genai
