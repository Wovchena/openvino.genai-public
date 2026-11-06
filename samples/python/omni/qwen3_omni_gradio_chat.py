#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Omni voice chat served through a Gradio web UI.

Preview: the Qwen3-Omni API (OmniPipeline and related types) is a preview feature
and is subject to change in future releases.

The browser records the question from the microphone, the model answers with text and
speech, and the speech plays back as it is produced: GenerationConfig.stream_text2speech
feeds thinker tokens to the talker while text decoding is still running, so the first
audio chunk arrives long before the full answer is decoded.

Usage:
    python qwen3_omni_gradio_chat.py <MODEL_DIR>
"""

import argparse
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import librosa
import numpy as np
import openvino_genai
from openvino import Tensor

# Qwen3-Omni audio encoder consumes 16kHz mono; the talker emits 24kHz mono PCM.
INPUT_SAMPLE_RATE = 16000
SPEECH_SAMPLE_RATE = 24000
# Gradio plays a streamed audio output smoothly when chunks are at least a second long.
PLAYBACK_CHUNK_SAMPLES = SPEECH_SAMPLE_RATE
VOICE_PLACEHOLDER = "🎤 voice question"
SERVER_PORT = 7860

AudioChunk = tuple[int, np.ndarray]
# Generation thread -> UI generator hand-off. None marks the end of generation.
GenerationEvent = tuple[str, str | np.ndarray] | None


def to_model_audio(recording: AudioChunk) -> Tensor:
    """Convert a Gradio (sample_rate, samples) recording to the float32 16kHz mono tensor the model expects."""
    sample_rate, samples = recording
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate != INPUT_SAMPLE_RATE:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=INPUT_SAMPLE_RATE)
    return Tensor(samples)


def to_playback_chunk(samples: np.ndarray) -> AudioChunk:
    # Browsers play int16 PCM; the talker emits float32 in [-1, 1].
    return SPEECH_SAMPLE_RATE, (np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)


class QueueSpeechStreamer(openvino_genai.OmniSpeechStreamerBase):
    """Forward talker audio chunks to the UI generator and stop the talker when the client is gone."""

    def __init__(self, events: queue.Queue, stop_requested: threading.Event) -> None:
        super().__init__()
        self.events = events
        self.stop_requested = stop_requested

    def write(self, audio_chunk: Tensor) -> openvino_genai.StreamingStatus:
        if self.stop_requested.is_set():
            return openvino_genai.StreamingStatus.STOP
        # The tensor is valid only for the duration of this call; np.array copies.
        self.events.put(("audio", np.array(audio_chunk.data, dtype=np.float32).reshape(-1)))
        return openvino_genai.StreamingStatus.RUNNING

    def end(self) -> None:
        pass


class OmniVoiceChat:
    """Owns the pipeline and the configs; one instance serves all browser sessions."""

    def __init__(self, model_dir: str, thinker_device: str, talker_device: str) -> None:
        # Compose OmniPipeline from its two stages so each can run on its own device.
        # On a shared GPU the thinker, which runs far ahead of what speech needs, would otherwise
        # monopolise the device and the first audio chunk would arrive only after the text is
        # complete. The scheduling priority hint is a GPU plugin property; the CPU plugin rejects it.
        shared_gpu = thinker_device == talker_device and thinker_device.startswith("GPU")
        thinker_properties = {"MODEL_PRIORITY": "LOW"} if shared_gpu else {}
        talker_properties = {"MODEL_PRIORITY": "HIGH"} if shared_gpu else {}
        vlm = openvino_genai.VLMPipeline(model_dir, thinker_device, **thinker_properties)
        talker = openvino_genai.Talker(model_dir, talker_device, **talker_properties)
        self.pipe = openvino_genai.OmniPipeline(vlm, talker)

        self.text_config = openvino_genai.GenerationConfig()
        self.text_config.max_new_tokens = 256
        # Stream thinker output to the talker so speech starts before the text is complete.
        # Requires return_audio below.
        self.text_config.stream_text2speech = True

        self.talker_speech_config = openvino_genai.OmniTalkerSpeechConfig(model_dir)
        self.talker_speech_config.return_audio = True
        # Codec frames per streamed chunk. The per-frame talker + code-predictor cost is fixed;
        # code2wav and its warm-up trim are paid once per chunk, so larger chunks raise speech
        # throughput at the cost of first-chunk latency. 8 is ~5% faster than the default 4.
        self.talker_speech_config.audio_chunk_frames = 8

    def respond(
        self,
        prompt: str,
        recording: AudioChunk | None,
        history: list[dict[str, str]],
        chat_display: list[dict[str, str]],
    ) -> Iterator[tuple]:
        """Run one chat turn, yielding (chat_display, speech chunk, history, prompt box, microphone) updates.

        Outputs that don't change in a given update are gr.skip(). history is what the model sees;
        chat_display is what the browser shows. They differ only in how a voice-only question is rendered.
        """
        if not prompt and recording is None:
            raise gr.Error("Record a question or type one.")

        history = history + [{"role": "user", "content": prompt}]
        chat_display = chat_display + [{"role": "user", "content": prompt or VOICE_PLACEHOLDER}]
        answer = ""

        # Every update is a new list: Gradio serializes a yielded value right away, but a value
        # mutated after the yield would still change what any other consumer has already received.
        def display_with_answer() -> list[dict[str, str]]:
            return chat_display + [{"role": "assistant", "content": answer}]

        # Clear the inputs right away so the user sees the turn was accepted.
        yield display_with_answer(), gr.skip(), history, "", None

        events: queue.Queue[GenerationEvent] = queue.Queue()
        stop_requested = threading.Event()

        def text_streamer(subword: str) -> openvino_genai.StreamingStatus:
            if stop_requested.is_set():
                return openvino_genai.StreamingStatus.STOP
            events.put(("text", subword))
            return openvino_genai.StreamingStatus.RUNNING

        audios = [] if recording is None else [to_model_audio(recording)]
        pending_samples: list[np.ndarray] = []
        pending_sample_count = 0

        # The pipeline runs on a worker thread so the UI generator can forward chunks as they
        # arrive. The executor's exit waits for generate() to finish, which keeps the single
        # pipeline instance from being entered by the next turn while this one still runs.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.pipe.generate,
                openvino_genai.ChatHistory(history),
                audios=audios,
                text_config=self.text_config,
                talker_speech_config=self.talker_speech_config,
                streamer=text_streamer,
                speech_streamer=QueueSpeechStreamer(events, stop_requested),
            )
            future.add_done_callback(lambda _: events.put(None))
            try:
                while (event := events.get()) is not None:
                    kind, payload = event
                    if kind == "text":
                        answer += payload
                        yield display_with_answer(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
                        continue
                    pending_samples.append(payload)
                    pending_sample_count += payload.size
                    if pending_sample_count >= PLAYBACK_CHUNK_SAMPLES:
                        yield (
                            gr.skip(),
                            to_playback_chunk(np.concatenate(pending_samples)),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                        )
                        pending_samples.clear()
                        pending_sample_count = 0
            finally:
                # Reached on normal completion and when the browser disconnects mid-turn; in the
                # latter case the streamers return STOP on their next call.
                stop_requested.set()
        decoded_results = future.result()

        if pending_samples:
            yield gr.skip(), to_playback_chunk(np.concatenate(pending_samples)), gr.skip(), gr.skip(), gr.skip()
        # The returned text is authoritative: stop strings can rewind streamed tokens.
        answer = decoded_results.texts[0]
        history = history + [{"role": "assistant", "content": answer}]
        yield display_with_answer(), gr.skip(), history, gr.skip(), gr.skip()


def build_demo(chat: OmniVoiceChat) -> gr.Blocks:
    with gr.Blocks(title="Qwen3-Omni voice chat") as demo:
        gr.Markdown(
            "# Qwen3-Omni voice chat\n"
            "Record a question (or type one) and press **Ask**. Speech playback starts while the answer is still being generated."
        )
        history = gr.State([])
        chat_display = gr.Chatbot(label="Conversation")
        speech_output = gr.Audio(label="Answer speech", streaming=True, autoplay=True, interactive=False)
        with gr.Row():
            microphone = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Question (voice)")
            prompt = gr.Textbox(label="Question (text)", placeholder="Optional text to send along with the recording")
        ask = gr.Button("Ask", variant="primary")

        inputs = [prompt, microphone, history, chat_display]
        outputs = [chat_display, speech_output, history, prompt, microphone]
        ask.click(chat.respond, inputs, outputs)
        prompt.submit(chat.respond, inputs, outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-Omni voice chat served through a Gradio web UI")
    parser.add_argument("model_dir", help="Path to the OpenVINO model directory")
    args = parser.parse_args()

    # Gap-free playback needs the talker to produce audio faster than real time, so it gets the
    # GPU. The thinker shares it: while both run, speech throughput drops, but on a device with
    # headroom that is the setup where speech starts earliest.
    demo = build_demo(OmniVoiceChat(args.model_dir, thinker_device="GPU", talker_device="CPU"))
    # One pipeline instance: run one turn at a time, queue the rest.
    demo.queue(default_concurrency_limit=1)
    # Browsers allow microphone capture only on https:// or on localhost. Reach a remote server over
    # `ssh -L 7860:localhost:7860 <host>` and open http://localhost:7860, or front it with TLS.
    demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT)


if __name__ == "__main__":
    main()
