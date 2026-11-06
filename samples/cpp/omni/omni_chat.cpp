// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"
#include "openvino/runtime/properties.hpp"
#include "speech_generation/audio_utils.hpp"
#include "visual_language_chat/load_image.hpp"
#include "visual_language_chat/load_video.hpp"

namespace {
constexpr uint32_t SPEECH_SAMPLE_RATE = 24000;  // Qwen3-Omni speech output is 24kHz mono PCM.

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

// Save the first waveform of a result to a WAV file. Speech output is optional (talker mode),
// so callers pass results whose waveforms may be empty.
void save_speech(const ov::genai::OmniDecodedResults& results, const std::string& file_name) {
    if (results.speech_result.waveforms.empty()) {
        return;
    }
    const ov::Tensor& waveform = results.speech_result.waveforms[0];
    utils::audio::save_to_wav(waveform.data<const float>(),
                              waveform.get_size(),
                              file_name,
                              waveform.get_element_type().bitwidth(),
                              SPEECH_SAMPLE_RATE);
}
}  // namespace

// Receives audio chunks as the talker produces them and reports the delay between the
// generate() call and the first chunk.
class ChunkPrinter : public ov::genai::OmniSpeechStreamerBase {
public:
    void start() {
        m_start = std::chrono::steady_clock::now();
        m_first_chunk_latency.reset();
        m_chunk_count = 0;
        m_sample_count = 0;
    }

    ov::genai::StreamingStatus write(const ov::Tensor& audio_chunk) override {
        if (!m_first_chunk_latency.has_value()) {
            m_first_chunk_latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - m_start);
            std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
            std::cout << "\n[Time to first audio chunk: " << m_first_chunk_latency->count() << " ms]" << std::endl;
        }
        ++m_chunk_count;
        m_sample_count += audio_chunk.get_size();
        return ov::genai::StreamingStatus::STOP;
    }

    void end() override {
        const auto total = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - m_start);
        std::cout << "\n[Streamed " << m_chunk_count << " chunks, " << m_sample_count << " samples at 24kHz in "
                  << total.count() << " ms]" << std::endl;
    }

private:
    std::chrono::steady_clock::time_point m_start;
    std::optional<std::chrono::milliseconds> m_first_chunk_latency;
    size_t m_chunk_count = 0;
    size_t m_sample_count = 0;
};

int main(int argc, char* argv[]) try {
    // if (argc != 5) {
    //     throw std::runtime_error(std::string{"Usage: "} + argv[0] +
    //                              " <MODEL_DIR> <IMAGE_FILE_OR_DIR> <AUDIO_FILE> <VIDEO_FILE>");
    // }

    const std::filesystem::path models_path = argv[1];

    // Two configs: text_config drives the thinker text decode, talker_speech_config drives
    // the talker + speech output. Speech output is hardcoded on here to show the multimodal
    // path. Set talker_speech_config.return_audio = false to get text-only responses.
    ov::genai::GenerationConfig text_config;
    text_config.max_new_tokens = 256;
    text_config.stream_text2speech = true;
    // 54808 ms - 100, gpu, cpu, true
    // 58524 ms - 100, gpu, cpu, false
    // 132741 ms - 256, gpu, cpu, false
    // 123998 ms - 256, gpu, cpu, true
    // 73178 ms - 128, gpu, cpu, false
    // 68508 ms - 128, gpu, cpu, true
    // 30888 ms - 50, gpu, cpu, true
    // 33121 ms - 50, gpu, cpu, false

    // 61875 ms - 100, cpu, gpu, false
    // 58115 ms - 100, cpu, gpu, true

    // [Time to first audio chunk: 10930 ms] - 100
    // [Time to first audio chunk: 4142 ms] - 100
    // [Time to first audio chunk: 4185 ms] - 256
    // [Time to first audio chunk: 22501 ms] - 256
    // [Time to first audio chunk: 12910 ms] - 128
    // [Time to first audio chunk: 4477 ms] - 128
    // [Time to first audio chunk: 4256 ms] - 50
    // [Time to first audio chunk: 7199 ms] - 50

    ov::genai::OmniTalkerSpeechConfig talker_speech_config(models_path);
    talker_speech_config.return_audio = true;
    // Leaving talker_speech_config.speaker empty selects the model's default voice. Available
    // voices vary by checkpoint.

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    ov::Tensor audio_tensor = utils::audio::read_wav_as_tensor(argv[3]);
    std::vector<ov::Tensor> audios = {std::move(audio_tensor)};

    // Compose OmniPipeline from separate VLM and Talker components.
    // This allows independent device selection (e.g., VLM on GPU, Talker on CPU),
    // sharing a VLM base across pipelines, or injecting custom TalkerBase implementations.
    // For simpler use cases: ov::genai::OmniPipeline pipe(models_path, "CPU");

    auto vlm = std::make_shared<ov::genai::VLMPipeline>(models_path, "GPU");
    auto talker = std::make_shared<ov::genai::Talker>(models_path, "CPU");
    ov::genai::OmniPipeline pipe(vlm, talker);

    // Speaker API demo: list available speakers and demonstrate voice blending.
    std::vector<std::string> speakers = talker->list_speakers();
    std::cout << "\n=== Available Speakers ===\n";
    if (speakers.empty()) {
        std::cout << "No named speakers (model has single default voice)\n";
    } else {
        std::cout << "Found " << speakers.size() << " speakers:\n";
        for (const auto& name : speakers) {
            std::cout << "  - " << name << "\n";
        }
        std::cout << "\n";

        if (speakers.size() >= 2) {
            std::cout << "=== Voice Blending Demo ===\n";
            std::cout << "Blending " << speakers[0] << " + " << speakers[1] << " (50/50 mix)\n";
            ov::Tensor emb1 = talker->get_speaker_embedding(speakers[0]);
            ov::Tensor emb2 = talker->get_speaker_embedding(speakers[1]);
            ov::Tensor blended(emb1.get_element_type(), emb1.get_shape());
            const float* data1 = emb1.data<float>();
            const float* data2 = emb2.data<float>();
            float* blended_data = blended.data<float>();
            for (size_t i = 0; i < emb1.get_size(); ++i) {
                blended_data[i] = 0.5f * data1[i] + 0.5f * data2[i];
            }
            talker_speech_config.speaker = blended;
        }
    }

    ov::genai::ChatHistory history;
    // auto [video, video_metadata] = utils::load_video(argv[4]);
    std::vector<ov::Tensor> videos;
    // videos.push_back(std::move(video));
    std::vector<ov::genai::VideoMetadata> videos_metadata;
    // videos_metadata.push_back(std::move(video_metadata));
    auto chunk_printer = std::make_shared<ChunkPrinter>();

    std::string prompt;
    std::cout << "question:\n";
    std::getline(std::cin, prompt);

    history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
    chunk_printer->start();

    auto start = std::chrono::steady_clock::now();
    ov::genai::OmniDecodedResults decoded_results = pipe.generate(history,
                                                                   rgbs,
                                                                   videos,
                                                                   videos_metadata,
                                                                   audios,
                                                                   text_config,
                                                                   talker_speech_config,
                                                                   print_subword,
                                                                   chunk_printer
                                                                );
    auto end = std::chrono::steady_clock::now();
    std::cout << "Generation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
    history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
    save_speech(decoded_results, "output_audio_0.wav");

    std::cout << "\n----------\n"
                 "question:\n";
    for (size_t turn = 1; std::getline(std::cin, prompt); ++turn) {
        history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        // Media attaches to the turn it is supplied on, so later turns pass none at all and refer
        // back through the history rather than re-sending turn 1's tensors.
        std::vector<ov::Tensor> turn_images, turn_videos, turn_audios;
        std::vector<ov::genai::VideoMetadata> turn_videos_metadata;
        chunk_printer->start();
        decoded_results = pipe.generate(history,
                                        turn_images,
                                        turn_videos,
                                        turn_videos_metadata,
                                        turn_audios,
                                        text_config,
                                        talker_speech_config,
                                        print_subword,
                                        chunk_printer
                                    );
        history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
        save_speech(decoded_results, "output_audio_" + std::to_string(turn) + ".wav");

        std::cout << "\n----------\n"
                     "question:\n";
    }
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
