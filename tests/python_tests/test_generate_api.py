# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import openvino
import openvino_genai
import openvino_tokenizers
import optimum.intel
import pytest
import transformers
from list_test_models import models_list


@pytest.fixture(scope="module", params=models_list())
@functools.lru_cache(1)
def model_fixture(request):
    model_id, path = request.param
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    openvino.save_model(ov_tokenizer, path / "openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, path / "openvino_detokenizer.xml")
    model = optimum.intel.openvino.OVModelForCausalLM.from_pretrained(model_id, export=True, device='CPU', load_in_8bit=False)
    model.save_pretrained(path)
    return model_id, path, tokenizer, model

def run_hf_ov_genai_comparison(model_fixture, generation_config, prompt):
    import openvino_genai as ov_genai
    model_id, path, tokenizer, model = model_fixture

    generation_config_hf = generation_config.copy()
    # in OpenVINO GenAI this parameter is called stop_criteria,
    # while in HF it's called early_stopping. 
    # HF values True, False and "never" correspond to OV GenAI values "early", "heuristic" and "never"
    if generation_config_hf.get('stop_criteria'):
        generation_config_hf['early_stopping'] = stop_criteria_map()[generation_config_hf.pop('stop_criteria')]

    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
    hf_encoded_output = model.generate(encoded_prompt, **generation_config_hf)
    hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])

    device = 'CPU'
    pipe = ov_genai.LLMPipeline(path, device)
    
    ov_output = pipe.generate(prompt, **generation_config)
    if generation_config.get('num_return_sequences', 1) > 1:
        ov_output = ov_output[0]

    if hf_output != ov_output:
        print(f'hf_output: {hf_output}')
        print(f'ov_output: {ov_output}')

    assert hf_output == ov_output


def stop_criteria_map():
    return {"never": "never", "early": True, "heuristic": False}

test_cases = [
    (dict(max_new_tokens=20, do_sample=False), 'table is made of'),  # generation_config, prompt
    (dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=20, diversity_penalty=1.0), 'Alan Turing was a'),
    (dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=30, diversity_penalty=1.0), 'Alan Turing was a'),
    (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.0), 'table is made of'),
    (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.0), 'The Sun is yellow because'),
    (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.5), 'The Sun is yellow because'),
]
@pytest.mark.parametrize("generation_config,prompt", test_cases)
def test_greedy_decoding(model_fixture, generation_config, prompt):
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)


prompts = ['The Sun is yellow because', 'Alan Turing was a', 'table is made of']
@pytest.mark.parametrize("num_beam_groups", [2, 3, 8])
@pytest.mark.parametrize("group_size", [5, 3, 10])
@pytest.mark.parametrize("max_new_tokens", [20, 15])
@pytest.mark.parametrize("diversity_penalty", [1.0, 1.5])
@pytest.mark.parametrize("prompt", prompts)
def test_beam_search_decoding(model_fixture, num_beam_groups, group_size, 
                              max_new_tokens, diversity_penalty, prompt):
    generation_config = dict(
        num_beam_groups=num_beam_groups, 
        num_beams=num_beam_groups * group_size, 
        diversity_penalty=diversity_penalty, 
        num_return_sequences=num_beam_groups * group_size, 
        max_new_tokens=max_new_tokens, 
    )
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)


@pytest.mark.parametrize("stop_criteria", ["never", "early", "heuristic"])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.parametrize("max_new_tokens", [20, 40, 300])
def test_stop_criteria(model_fixture, stop_criteria, prompt, max_new_tokens):
    # todo: for long sentences early stop_criteria fails
    if (stop_criteria == 'early' and max_new_tokens >= 300):
        pytest.skip()
    generation_config = dict(
        num_beam_groups=2, 
        num_beams=2 * 3, 
        diversity_penalty=1.0, 
        num_return_sequences=2 * 3, 
        max_new_tokens=max_new_tokens, 
        stop_criteria=stop_criteria,
    )
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)


# test long sequences
@pytest.mark.parametrize("num_beam_groups", [2])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("max_new_tokens", [800, 2000])
@pytest.mark.parametrize("diversity_penalty", [1.0])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.skip  # will be enabled in nightly since are computationally expensive
def test_beam_search_long_sentences(model_fixture, num_beam_groups, group_size, 
                              max_new_tokens, diversity_penalty, prompt):
    generation_config = dict(
        num_beam_groups=num_beam_groups, 
        num_beams=num_beam_groups * group_size, 
        diversity_penalty=1.0, 
        num_return_sequences=num_beam_groups * group_size, 
        max_new_tokens=max_new_tokens, 
    )
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)


def user_defined_callback(subword):
    print(subword)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_callback_one_string(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    pipe.generate('', openvino_genai.GenerationConfig(), callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_callback_batch_fail(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], openvino_genai.GenerationConfig(), callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_callback_kwargs_one_string(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    pipe.generate('', max_new_tokens=10, streamer=callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_callback_kwargs_batch_fail(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], max_new_tokens=10, streamer=callback)


class Printer(openvino_genai.StreamerBase):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    def put(self, token_id):
        print(self.tokenizer.decode([token_id]))  # Incorrect way to print, but easy to implement
    def end(self):
        print('end')


def test_streamer_one_string(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    pipe.generate('', openvino_genai.GenerationConfig(), printer)


def test_streamer_batch_fail(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], openvino_genai.GenerationConfig(), printer)


def test_streamer_kwargs_one_string(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    pipe.generate('', do_sample=True, streamer=printer)


def test_streamer_kwargs_batch_fail(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe.generate('', num_beams=2, streamer=printer)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_operator_wit_callback_one_string(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    pipe('', openvino_genai.GenerationConfig(), callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_operator_wit_callback_batch_fail(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    with pytest.raises(RuntimeError):
        pipe(['1', '2'], openvino_genai.GenerationConfig(), callback)


def test_perator_wit_streamer_kwargs_one_string(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    pipe('', do_sample=True, streamer=printer)


def test_erator_wit_streamer_kwargs_batch_fail(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe('', num_beams=2, streamer=printer)