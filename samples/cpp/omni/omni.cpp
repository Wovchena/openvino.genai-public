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
// Let users implement generate() token by token - from llama.cpp
// Return dict[str->[tensor, float, str]] (transformers) vs inheritance (my idea) vs optional members (llama.cpp) in return type
// streaming https://github.com/OpenBMB/MiniCPM-V
// try AnytoAnyPipeline with omni mdels
// find the model that generates text and images
// ask LLMs about multimodal and omni
// https://github.com/GeeeekExplorer/nano-vllm - no omni support
// https://docs.ollama.com/capabilities/vision - no omni
// vLLM-omni has prompt_embeds and additional_information members but could just be additional_information which is Any
// What model output to append to history
// Tokenizer + embedding layer is just one modality example
// What is the model state in terms of modalities
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
        VLMPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties = {},
        const ov::genai::GenerationConfig& generation_config = {}
    );
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VLMPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        Properties&&... properties)
        : VLMPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    ContinuousBatchingPipeline(const std::filesystem::path& models_path,
                               const SchedulerConfig& scheduler_config,
                               const std::string& device,
                               const ov::AnyMap& properties = {},
                               const ov::AnyMap& tokenizer_properties = {},
                               const ov::AnyMap& vision_encoder_properties = {});
    ContinuousBatchingPipeline(
        const std::filesystem::path& models_path,
        const ov::genai::Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );
    ContinuousBatchingPipeline(
        const std::string& model_str,
        const ov::Tensor& weights_tensor,
        const ov::genai::Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties = {},
        const ov::genai::GenerationConfig& generation_config = {}
    );
    ContinuousBatchingPipeline(
        const ModelsMap& models_map,
        const ov::genai::Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        std::optional<std::filesystem::path> embedder_config_dir_path = std::nullopt,
        const ov::AnyMap& properties = {},
        const ov::genai::GenerationConfig& generation_config = {}
    );

    GenerationHandle add_request(uint64_t request_id, const ov::Tensor& input_ids, const ov::genai::GenerationConfig& sampling_params);
    GenerationHandle add_request(uint64_t request_id, const std::string& prompt, const ov::genai::GenerationConfig& sampling_params);
    GenerationHandle add_request(uint64_t request_id, const std::string& prompt, const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos, const ov::genai::GenerationConfig& sampling_params);
    GenerationHandle add_request(uint64_t request_id, const std::string& prompt, const std::vector<ov::Tensor>& images, const ov::genai::GenerationConfig& sampling_params);

    void step();

    bool has_non_finished_requests();

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
    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    );
    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    );
    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::Tensor& image,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    );
    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    );
    template <typename... Properties>
    util::EnableIfAllStringAny<VLMDecodedResults, Properties...> generate(
        const std::string& prompt,
        Properties&&... properties
    ) {
        return generate(
            prompt, AnyMap{std::forward<Properties>(properties)...}
        );
    }
    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    );
    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    );
    VLMDecodedResults generate(
        const ChatHistory& history,
        const ov::Tensor& image,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    );
    VLMDecodedResults generate(
        const ChatHistory& history,
        const ov::AnyMap& config_map
    );
    template <typename... Properties>
    util::EnableIfAllStringAny<VLMDecodedResults, Properties...> generate(
        const ChatHistory& history,
        Properties&&... properties
    ) {
        return generate(
            history, AnyMap{std::forward<Properties>(properties)...}
        );
    }
    std::vector<EncodedGenerationResult> generate(const std::vector<ov::Tensor>& input_ids, const std::vector<ov::genai::GenerationConfig>& sampling_params, const ov::genai::StreamerVariant& streamer=std::monostate{});
    std::vector<GenerationResult> generate(const std::vector<std::string>& prompts, const std::vector<ov::genai::GenerationConfig>& sampling_params, const ov::genai::StreamerVariant& streamer=std::monostate{});

    std::vector<GenerationResult> generate(
        const std::vector<ChatHistory>& histories,
        const std::vector<ov::genai::GenerationConfig>& sampling_params,
        const ov::genai::StreamerVariant& streamer=std::monostate{});

    std::vector<VLMDecodedResults> generate(
             const std::vector<std::string>& prompts,
             const std::vector<std::vector<ov::Tensor>>& images,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer=std::monostate{});

    std::vector<VLMDecodedResults> generate(
        const std::vector<std::string>& prompts,
        const std::vector<std::vector<ov::Tensor>>& images,
        const std::vector<std::vector<ov::Tensor>>& videos,
        const std::vector<GenerationConfig>& sampling_params,
        const StreamerVariant& streamer=std::monostate{});

    std::vector<VLMDecodedResults> generate(
        const std::vector<ChatHistory>& histories,
        const std::vector<std::vector<ov::Tensor>>& images,
        const std::vector<GenerationConfig>& sampling_params,
        const StreamerVariant& streamer=std::monostate{});

    std::vector<VLMDecodedResults> generate(
        const std::vector<ChatHistory>& histories,
        const std::vector<std::vector<ov::Tensor>>& images,
        const std::vector<std::vector<ov::Tensor>>& videos,
        const std::vector<GenerationConfig>& sampling_params,
        const StreamerVariant& streamer=std::monostate{});
        EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config = std::nullopt,
        StreamerVariant streamer=std::monostate()
    );

    /**
    * @brief Low level generate to be called with already encoded input_ids tokens.
    * Streamer cannot be used for multibatch inputs.
    *
    * @param input_ids or pair of (input_ids, attentino_mask) encoded input prompt tokens
    * @param generation config params
    * @return EncodedResults a structure with resulting tokens and scores
    * @throws Exception if the stremaer is set for inputs_ids with multiple batches
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<EncodedResults, Properties...> generate(
            const EncodedInputs& inputs,
            Properties&&... properties) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }
    EncodedResults generate(const EncodedInputs& inputs, const ov::AnyMap& config_map);

    void set_chat_template(const std::string& new_template);
    ov::genai::Tokenizer get_tokenizer() const;
    // set_tokenizer?

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
