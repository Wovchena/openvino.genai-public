// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cxxopts.hpp>

#include "continuous_batching_pipeline.hpp"

void print_generation_result(const GenerationResult& generation_result) {
    for (size_t output_id = 0; output_id < generation_result.m_generation_ids.size(); ++output_id) {
        std::cout << "Answer " << output_id << " (" << generation_result.m_scores[output_id] << ") : " << generation_result.m_generation_ids[output_id] << std::endl;
    }
}

ov::genai::GenerationConfig beam_search() {
    ov::genai::GenerationConfig beam_search;
    beam_search.num_beams = 4;
    beam_search.num_return_sequences = 3;
    beam_search.num_beam_groups = 2;
    beam_search.max_new_tokens = 100;
    beam_search.diversity_penalty = 2.0f;
    return beam_search;
}

ov::genai::GenerationConfig greedy() {
    ov::genai::GenerationConfig greedy_params;
    greedy_params.temperature = 0.0f;
    greedy_params.ignore_eos = true;
    greedy_params.num_return_sequences = 1;
    greedy_params.repetition_penalty = 3.0f;
    greedy_params.presence_penalty = 0.1f;
    greedy_params.frequency_penalty = 0.01f;
    greedy_params.max_new_tokens = 30;
    return greedy_params;
}

ov::genai::GenerationConfig multinomial() {
    ov::genai::GenerationConfig multinomial;
    multinomial.do_sample = true;
    multinomial.temperature = 0.9f;
    multinomial.top_p = 0.9f;
    multinomial.top_k = 20;
    multinomial.num_return_sequences = 3;
    multinomial.presence_penalty = 0.01f;
    multinomial.frequency_penalty = 0.1f;
    multinomial.min_new_tokens = 15;
    multinomial.max_new_tokens = 30;
    return multinomial;
}

int main(int argc, char* argv[]) try {
    // Command line options

    cxxopts::Options options("accuracy_sample", "Help command");

    options.add_options()
    ("n,num_prompts", "A number of prompts", cxxopts::value<size_t>()->default_value("1"))
    ("dynamic_split_fuse", "Whether to use dynamic split-fuse or vLLM scheduling", cxxopts::value<bool>()->default_value("false"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const size_t num_prompts = result["num_prompts"].as<size_t>();
    const bool dynamic_split_fuse = result["dynamic_split_fuse"].as<bool>();
    const std::string models_path = result["model"].as<std::string>();

    // create dataset

    std::vector<std::string> prompt_examples = {
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada",
        "What is OpenVINO?",
    };

    std::vector<ov::genai::GenerationConfig> sampling_params_examples {
        beam_search(),
        greedy(),
        multinomial(),
    };

    std::vector<std::string> prompts(num_prompts);
    std::vector<ov::genai::GenerationConfig> sampling_params(num_prompts);

    for (size_t request_id = 0; request_id < num_prompts; ++request_id) {
        prompts[request_id] = prompt_examples[request_id % prompt_examples.size()];
        sampling_params[request_id] = sampling_params_examples[request_id % sampling_params_examples.size()];
    }

    // Perform the inference
    
    SchedulerConfig scheduler_config {
        // batch size
        .max_num_batched_tokens = 32,
        // cache params
        .num_kv_blocks = 364,
        .block_size = 32,
        // mode - vLLM or dynamic_split_fuse
        .dynamic_split_fuse = dynamic_split_fuse,
        // vLLM specific params
        .max_num_seqs = 2,
    };

    ContinuousBatchingPipeline pipe(models_path, scheduler_config);
    std::vector<GenerationResult> generation_results = pipe.generate(prompts, sampling_params);

    for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
        const GenerationResult & generation_result = generation_results[request_id];
        std::cout << "Question: " << prompts[request_id] << std::endl;
        switch (generation_result.m_status)
        {
        case GenerationStatus::FINISHED:
            print_generation_result(generation_result);
            break;
        case GenerationStatus::IGNORED:
            std::cout << "Request was ignored due to lack of memory." <<std::endl;
            if (generation_result.m_generation_ids.size() > 0) {
                std::cout << "Partial result:" << std::endl;
                print_generation_result(generation_result);
            }
            break;
        case GenerationStatus::DROPPED_BY_PIPELINE:
            std::cout << "Request was aborted." <<std::endl;
            if (generation_result.m_generation_ids.size() > 0) {
                std::cout << "Partial result:" << std::endl;
                print_generation_result(generation_result);
            }
            break;   
        default:
            break;
        }
        std::cout << std::endl;
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
