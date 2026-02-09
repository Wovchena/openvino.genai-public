// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Sample demonstrating the Omni Pipeline API for multimodal models
 * 
 * This sample shows how to use OmniPipeline to:
 * - Process text, images, and audio inputs
 * - Generate text and audio outputs
 * - Maintain conversational context across modalities
 */

#include <openvino/genai/omni/pipeline.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

// Simple audio file loading (WAV format assumed)
std::vector<float> load_audio(const std::string& filename) {
    // This is a placeholder. In a real implementation, use a library like
    // libsndfile or implement WAV parsing
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open audio file: " + filename);
    }
    
    // Skip WAV header (44 bytes for standard WAV)
    file.seekg(44);
    
    // Read audio data
    std::vector<float> samples;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        // Convert int16 to float [-1.0, 1.0]
        samples.push_back(static_cast<float>(sample) / 32768.0f);
    }
    
    return samples;
}

// Simple image loading (placeholder)
ov::Tensor load_image(const std::string& filename) {
    // This is a placeholder. In a real implementation, use OpenCV or similar
    // For demonstration, return an empty tensor
    // Real implementation would decode image and return RGB uint8 tensor
    throw std::runtime_error("Image loading not implemented in this example");
}

// Save audio to WAV file
void save_audio(const std::vector<float>& samples, size_t sample_rate, const std::string& filename) {
    // This is a placeholder. In a real implementation, write proper WAV format
    std::cout << "Audio would be saved to: " << filename << std::endl;
    std::cout << "Samples: " << samples.size() << ", Sample rate: " << sample_rate << " Hz" << std::endl;
}

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

void demonstrate_text_to_text(ov::genai::OmniPipeline& pipe) {
    std::cout << "\n=== Text-to-Text Generation ===\n";
    
    ov::genai::OmniGenerationConfig config;
    config.max_new_tokens = 100;
    config.output_modality = "text";
    
    auto result = pipe.generate(
        "What is OpenVINO?",
        config,
        print_subword
    );
    
    std::cout << "\n\nPerformance metrics:\n";
    std::cout << "  TTFT: " << result.omni_perf_metrics.get_ttft().mean << " ms\n";
    std::cout << "  TPOT: " << result.omni_perf_metrics.get_tpot().mean << " ms\n";
}

void demonstrate_text_to_speech(ov::genai::OmniPipeline& pipe) {
    std::cout << "\n=== Text-to-Speech Generation ===\n";
    
    ov::genai::OmniGenerationConfig config;
    config.output_modality = "audio";
    config.voice = "alloy";
    config.audio_sample_rate = 24000;
    config.max_new_tokens = 0;  // Not applicable for audio-only output
    
    auto result = pipe.generate(
        "Hello! This is a demonstration of text to speech conversion.",
        config,
        std::monostate{}
    );
    
    if (result.audio.has_value()) {
        save_audio(result.audio.value(), result.audio_sample_rate, "output.wav");
        std::cout << "Audio generated: " << result.audio->size() / result.audio_sample_rate 
                  << " seconds\n";
    }
    
    std::cout << "Audio encoding duration: " 
              << result.omni_perf_metrics.audio_encoding_duration.mean << " ms\n";
}

void demonstrate_audio_transcription(ov::genai::OmniPipeline& pipe, const std::string& audio_file) {
    std::cout << "\n=== Audio Transcription ===\n";
    
    try {
        auto audio_samples = load_audio(audio_file);
        
        ov::genai::OmniGenerationConfig config;
        config.max_new_tokens = 200;
        config.output_modality = "text";
        
        std::cout << "Transcription: ";
        auto result = pipe.generate(
            "Transcribe the following audio:",
            audio_samples,
            config,
            print_subword
        );
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cout << "Audio transcription skipped: " << e.what() << "\n";
    }
}

void demonstrate_image_understanding(ov::genai::OmniPipeline& pipe, const std::string& image_file) {
    std::cout << "\n=== Image Understanding ===\n";
    
    try {
        auto image = load_image(image_file);
        
        ov::genai::OmniGenerationConfig config;
        config.max_new_tokens = 150;
        config.output_modality = "text";
        
        std::cout << "Description: ";
        auto result = pipe.generate(
            "Describe what you see in this image in detail.",
            image,
            config,
            print_subword
        );
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cout << "Image understanding skipped: " << e.what() << "\n";
    }
}

void demonstrate_multimodal_chat(ov::genai::OmniPipeline& pipe) {
    std::cout << "\n=== Multimodal Conversational Chat ===\n";
    std::cout << "Enter 'exit' to quit the chat.\n";
    std::cout << "Commands:\n";
    std::cout << "  /voice <name>  - Set voice for audio output (e.g., alloy, echo, nova)\n";
    std::cout << "  /audio         - Toggle audio output\n";
    std::cout << "  /help          - Show this help\n\n";
    
    pipe.start_chat("You are a helpful AI assistant with multimodal capabilities.");
    
    ov::genai::OmniGenerationConfig config;
    config.max_new_tokens = 150;
    config.output_modality = "text";
    bool audio_output = false;
    
    std::string prompt;
    while (true) {
        std::cout << "\nYou: ";
        std::getline(std::cin, prompt);
        
        if (prompt == "exit") {
            break;
        } else if (prompt.rfind("/voice ", 0) == 0) {
            config.voice = prompt.substr(7);
            std::cout << "Voice set to: " << config.voice << "\n";
            continue;
        } else if (prompt == "/audio") {
            audio_output = !audio_output;
            config.output_modality = audio_output ? "text+audio" : "text";
            std::cout << "Audio output: " << (audio_output ? "enabled" : "disabled") << "\n";
            continue;
        } else if (prompt == "/help") {
            std::cout << "Commands:\n";
            std::cout << "  /voice <name>  - Set voice for audio output\n";
            std::cout << "  /audio         - Toggle audio output\n";
            std::cout << "  /help          - Show this help\n";
            continue;
        }
        
        std::cout << "Assistant: ";
        auto result = pipe.generate(
            prompt,
            config,
            print_subword
        );
        std::cout << "\n";
        
        if (result.audio.has_value()) {
            std::cout << "[Audio generated: " 
                      << result.audio->size() / result.audio_sample_rate 
                      << " seconds]\n";
        }
    }
    
    pipe.finish_chat();
}

int main(int argc, char* argv[]) try {
    if (argc < 2 || argc > 4) {
        std::cout << "Usage: " << argv[0] << " <MODEL_DIR> [IMAGE_FILE] [AUDIO_FILE] [DEVICE]\n";
        std::cout << "\nExamples:\n";
        std::cout << "  " << argv[0] << " ./gpt4o-model\n";
        std::cout << "  " << argv[0] << " ./gpt4o-model image.jpg\n";
        std::cout << "  " << argv[0] << " ./gpt4o-model image.jpg audio.wav\n";
        std::cout << "  " << argv[0] << " ./gpt4o-model image.jpg audio.wav GPU\n";
        return EXIT_SUCCESS;
    }
    
    std::string model_path = argv[1];
    std::string image_file = argc >= 3 ? argv[2] : "";
    std::string audio_file = argc >= 4 ? argv[3] : "";
    std::string device = argc >= 5 ? argv[4] : "CPU";
    
    std::cout << "Loading Omni model from: " << model_path << "\n";
    std::cout << "Device: " << device << "\n";
    
    ov::AnyMap properties;
    if (device == "GPU") {
        // Enable model caching for GPU
        properties.insert({ov::cache_dir("omni_cache")});
    }
    
    ov::genai::OmniPipeline pipe(model_path, device, properties);
    std::cout << "Model loaded successfully!\n";
    
    // Demonstrate different capabilities
    demonstrate_text_to_text(pipe);
    demonstrate_text_to_speech(pipe);
    
    if (!image_file.empty() && std::filesystem::exists(image_file)) {
        demonstrate_image_understanding(pipe, image_file);
    }
    
    if (!audio_file.empty() && std::filesystem::exists(audio_file)) {
        demonstrate_audio_transcription(pipe, audio_file);
    }
    
    // Interactive chat
    std::cout << "\n" << std::string(50, '=') << "\n";
    demonstrate_multimodal_chat(pipe);
    
    return EXIT_SUCCESS;
    
} catch (const std::exception& error) {
    try {
        std::cerr << "Error: " << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Unknown error occurred\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
