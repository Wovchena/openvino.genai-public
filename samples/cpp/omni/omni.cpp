// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/runtime/tensor.hpp>
#include <openvino/runtime/properties.hpp>

#include <filesystem>

namespace ov::genai {
// AnyToAnyPipeline - transformers' name, Any2AnyPipeline - adapt transformers' name to GenAi's iamge generation. TODO: other names. If it's going to be a base class, call it Pipeline
// image/video_generation building blocks and rag are ignored

// generate return value = what to return from get_generation_config()
// Avoid loading needless pieces
class AnyToAnyPipeline {
public:
    explicit AnyToAnyPipeline(const std::filesystem::path& models_dir) {}
    AnyToAnyPipeline(const std::filesystem::path& models_dir, const std::string& device, const AnyMap& properties = {}) {}
    AnyToAnyPipeline(
        const std::filesystem::path& models_dir, const std::string& device, const Config& config, const ov::AnyMap& properties = {}
    ) {}
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    AnyToAnyPipeline(const std::filesystem::path& models_dir, const std::string& device, Properties&&... properties)
        : AnyToAnyPipeline(models_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    const VideoGenerationConfig get_generation_config() const {}
    void set_generation_config(const VideoGenerationConfig& generation_config) {}
    SpeechGenerationConfig get_generation_config() const;
    void set_generation_config(const SpeechGenerationConfig& new_config) {}

    AnyToAnyPipeline clone();
    void reshape(const int64_t num_videos_per_prompt, const int64_t num_frames, const int64_t height, const int64_t width, const float guidance_scale) {}
    void compile(const std::string& device, const ov::AnyMap& properties = {}) {}
    template <typename... Properties> ov::util::EnableIfAllStringAny<void, Properties...> compile(
        const std::string& device, Properties&&... properties
    ) {return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});}
    void compile(const std::string& text_encode_device, const std::string&  _device, const std::string& vae_device, const ov::AnyMap& properties = {}) {}
    template <typename... Properties> ov::util::EnableIfAllStringAny<void, Properties...> compile(
        const std::string& text_encode_device, const std::string& denoise_device, const std::string& vae_device, Properties&&... properties
    ) {return compile(text_encode_device, denoise_device, vae_device, ov::AnyMap{std::forward<Properties>(properties)...});}
    VideoGenerationResult generate(const std::string& positive_prompt, const ov::AnyMap& properties = {}) {}
    template <typename... Properties> ov::util::EnableIfAllStringAny<VideoGenerationResult, Properties...> generate(
        const std::string& positive_prompt, Properties&&... properties
    ) {return generate(positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});}
    VideoGenerationResult decode(const ov::Tensor& latent) {}
    void export_model(const std::filesystem::path& blobs_dir) {}

    EmbeddingResults embed_documents(const std::vector<std::string>& texts);
    void start_embed_documents_async(const std::vector<std::string>& texts);
    EmbeddingResults wait_embed_documents();
    EmbeddingResult embed_query(const std::string& text);
    void start_embed_query_async(const std::string& text);
    EmbeddingResult wait_embed_query();
    std::vector<std::pair<size_t, float>> rerank(const std::string& query, const std::vector<std::string>& texts);
    void start_rerank_async(const std::string& query, const std::vector<std::string>& texts);
    std::vector<std::pair<size_t, float>> wait_rerank();

    Text2SpeechDecodedResults generate(
        const std::string& text, const ov::Tensor& speaker_embedding = ov::Tensor(), const ov::AnyMap& properties = {}
    ) {return generate(std::vector<std::string>{text}, speaker_embedding, properties);}
    Text2SpeechDecodedResults generate(
        const std::vector<std::string>& texts,
        const ov::Tensor& speaker_embedding = ov::Tensor(),
        const ov::AnyMap& properties = {}
    ) {}
    template <typename... Properties> Text2SpeechDecodedResults generate(
        const std::vector<std::string>& texts,
        const ov::Tensor& speaker_embedding = ov::Tensor(),
        Properties&&... properties
    ) {return generate(texts, speaker_embedding, ov::AnyMap{std::forward<Properties>(properties)...});}
};
}  // namespace ov::genai

int main() {}
