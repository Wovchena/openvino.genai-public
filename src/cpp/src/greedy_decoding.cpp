// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "generation_config_helper.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"

namespace {

void update_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask);
void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask, int64_t start_pos = 0);
ov::Tensor extend_attention(ov::Tensor attention_mask);

void update_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask) {
    const size_t batch_size = attention_mask.get_shape()[0];
    const size_t atten_length = attention_mask.get_shape()[1];
    position_ids.set_shape({batch_size, 1});

    for (size_t batch = 0; batch < batch_size; batch++) {
        int64_t* start = attention_mask.data<int64_t>() + batch * atten_length;
        position_ids.data<int64_t>()[batch] = std::accumulate(start, start + atten_length, 0);
    }
}

void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask, int64_t start_pos) {
    const size_t batch_size = attention_mask.get_shape()[0];
    const size_t seq_length = attention_mask.get_shape()[1];

    const int64_t* attention_mask_data = attention_mask.data<int64_t>();
    int64_t* position_ids_data = position_ids.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t sum = start_pos;
        for (size_t i = 0; i < seq_length; i++) {
            const size_t element_offset = batch * seq_length + i;
            position_ids_data[element_offset] = sum;
            if (attention_mask_data[element_offset] == 1) {
                sum += 1;
            }
        }
    }
}

ov::Tensor extend_attention(ov::Tensor attention_mask) {
    auto shape = attention_mask.get_shape();
    auto batch_size = shape[0];
    auto seq_len = shape[1];

    ov::Tensor new_atten_mask = ov::Tensor{attention_mask.get_element_type(), {batch_size, seq_len + 1}};
    auto old_data = attention_mask.data<int64_t>();
    auto new_data = new_atten_mask.data<int64_t>();
    for (size_t batch = 0; batch < batch_size; ++batch) {
        std::memcpy(new_data + batch * (seq_len + 1), old_data + batch * seq_len, seq_len * sizeof(int64_t));
        new_data[batch * (seq_len + 1) + seq_len] = 1;
    }
    return new_atten_mask;
}

}

namespace ov {

ov::EncodedResults greedy_decoding(ov::InferRequest& m_model_runner, 
                                       ov::Tensor input_ids, ov::Tensor attention_mask, ov::GenerationConfig generation_config, 
                                       std::shared_ptr<StreamerBase> streamer, bool is_chat_conversation) {
    
    ov::GenerationConfigHelper config_helper = generation_config;
    ov::Shape prompts_shape = input_ids.get_shape();
    size_t batch_size = prompts_shape[0];
    size_t prompt_len = prompts_shape[1];
    
    auto kv_cache_len = m_model_runner.query_state()[0].get_state().get_shape()[2];

    // todo: make this work even if position_ids are not specified
    auto position_ids = ov::Tensor{ov::element::i64, input_ids.get_shape()};
    initialize_position_ids(position_ids, attention_mask, kv_cache_len);

    ov::EncodedResults results;
    results.scores.resize(batch_size);
    results.tokens.resize(batch_size);
    std::fill(results.scores.begin(), results.scores.end(), 0);
    
    if (is_chat_conversation && kv_cache_len > 0) {
        auto attentions_mask_history = m_model_runner.get_tensor("attention_mask");

        size_t new_prompt_len = attention_mask.get_shape()[1];
        size_t context_len = attentions_mask_history.get_shape()[1];
        ov::Tensor new_attention_mask =  ov::Tensor{ov::element::i64, {1, context_len + new_prompt_len}};

        for (size_t i = 0; i < context_len; ++i) {
            auto r = attentions_mask_history.data<int64_t>()[i];
            new_attention_mask.data<int64_t>()[i] = attentions_mask_history.data<int64_t>()[i];
        }
        for (size_t i = context_len; i < context_len + new_prompt_len; ++i) {
            auto r = attention_mask.data<int64_t>()[i];
            new_attention_mask.data<int64_t>()[i] = attention_mask.data<int64_t>()[i - context_len];
        }
        m_model_runner.set_tensor("attention_mask", new_attention_mask);
    } else {
        m_model_runner.set_tensor("attention_mask", attention_mask);
    }

    auto atten_shape = attention_mask.get_shape();
    auto pos_shape = position_ids.get_shape();
    auto input_ids_shape = input_ids.get_shape();

    m_model_runner.set_tensor("input_ids", input_ids);
    m_model_runner.set_tensor("position_ids", position_ids);

    m_model_runner.get_tensor("beam_idx").set_shape({batch_size});
    auto beam_data = m_model_runner.get_tensor("beam_idx").data<int32_t>();
    std::iota(beam_data, beam_data + batch_size, 0);

    size_t max_tokens = config_helper.get_max_new_tokens(prompt_len);
    
    m_model_runner.infer();
    auto logits = m_model_runner.get_tensor("logits");
    ov::Shape logits_shape = logits.get_shape();
    size_t seq_len = logits_shape[1], vocab_size = logits_shape[2];
    m_model_runner.get_tensor("input_ids").set_shape({batch_size, 1});

    std::vector<int64_t> token_iter_results(batch_size);  // results of a single infer request
    std::vector<int> eos_met(batch_size, 0);  // use int because can not use std::all_of with vector<bool>
    for (size_t batch = 0; batch < batch_size; ++batch) {
        auto res = generate_utils::softmax(logits, batch);
        auto out_token = res.first;
        results.tokens[batch].emplace_back(res.first);
        results.scores[batch] += res.second;

        token_iter_results[batch] = out_token;
        eos_met[batch] = (out_token == generation_config.eos_token_id);
        m_model_runner.get_tensor("input_ids").data<int64_t>()[batch] = out_token;
    }
    if (streamer)
        streamer->put(token_iter_results[0]);

    bool all_are_eos = std::all_of(eos_met.begin(), eos_met.end(), [](int elem) { return elem == 1; });
    if (!generation_config.ignore_eos && all_are_eos)
        return results;
    
    for (size_t i = 0; i < max_tokens - 1; ++i) {
        update_position_ids(position_ids, m_model_runner.get_tensor("attention_mask"));
        m_model_runner.set_tensor("attention_mask", extend_attention(m_model_runner.get_tensor("attention_mask")));

        // todo: consider replacing with start_async and run callback right after that
        m_model_runner.infer();
        auto logits = m_model_runner.get_tensor("logits");
        ov::Shape logits_shape = logits.get_shape();
        size_t seq_len = logits_shape[1], vocab_size = logits_shape[2];
        
        std::vector<int64_t> token_iter_results(batch_size);  // results of a single infer request
        std::vector<int> eos_met(batch_size, 0);  // use int because can not use std::all_of with vector<bool>
        for (size_t batch = 0; batch < batch_size; ++batch) {

            auto res = ov::generate_utils::softmax(logits, batch);
            auto out_token = res.first;
            results.tokens[batch].emplace_back(res.first);
            results.scores[batch] += res.second;

            token_iter_results[batch] = out_token;
            eos_met[batch] = (out_token == generation_config.eos_token_id);

            m_model_runner.get_tensor("input_ids").data<int64_t>()[batch] = out_token;
        }
        if (streamer)
            streamer->put(token_iter_results[0]);

        // stop generation when EOS is met in all batches
        bool all_are_eos = std::all_of(eos_met.begin(), eos_met.end(), [](int elem) { return elem == 1; });
        if (!generation_config.ignore_eos && all_are_eos)
            break;
    }
    return results;
}

}