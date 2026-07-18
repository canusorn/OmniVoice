#!/usr/bin/env python3
import argparse
import logging
import os
from datetime import datetime

import gradio as gr
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.demo import build_demo
from omnivoice.utils.common import get_best_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("omnivoice-webui")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
REF_AUDIO_DIR = os.environ.get("REF_AUDIO_DIR", "/ref_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_AUDIO_DIR, exist_ok=True)


def save_audio_file(sampling_rate: int, waveform):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"omnivoice_{ts}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    sf.write(filepath, waveform, sampling_rate)
    logger.info(f"Saved: {filepath}")
    return filepath


def resolve_ref_audio(ref_audio):
    if isinstance(ref_audio, str) and ref_audio.startswith("ref://"):
        filename = ref_audio.removeprefix("ref://")
        path = os.path.join(REF_AUDIO_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"ref_audio file not found: {filename}")
        logger.info(f"Resolved ref:// -> {path}")
        return path
    return ref_audio


def make_generate_fn(model: OmniVoice):
    sampling_rate = model.sampling_rate

    def generate_fn(
        text, language, ref_audio, instruct,
        num_step, guidance_scale, denoise,
        speed, duration, preprocess_prompt, postprocess_output,
        mode, ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        try:
            ref_audio = resolve_ref_audio(ref_audio)
        except FileNotFoundError as e:
            return None, str(e)

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None
        kw = dict(text=text.strip(), language=lang, generation_config=gen_config)

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "Please upload a reference audio."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            import gc
            if hasattr(model, '_asr_pipe') and model._asr_pipe is not None:
                del model._asr_pipe
                model._asr_pipe = None
                torch.cuda.empty_cache()
                gc.collect()

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            logger.exception("Generation failed")
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype("int16")

        saved_path = save_audio_file(sampling_rate, waveform)

        return (sampling_rate, waveform), f"Done. Saved: {saved_path}"

    return generate_fn


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Web UI")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--load-asr", action="store_true")
    args = parser.parse_args()

    device = args.device or get_best_device()
    logger.info(f"Loading model from {args.model}, device={device} ...")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Reference audio directory: {REF_AUDIO_DIR}")

    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.float16,
        load_asr=args.load_asr,
        asr_model_name=args.asr_model,
    )

    asr_model = args.asr_model
    original_load_asr = model.load_asr_model
    model.load_asr_model = lambda model_name=asr_model: original_load_asr(model_name)

    logger.info("Model loaded successfully!")

    generate_fn = make_generate_fn(model)
    demo = build_demo(model, args.model, generate_fn=generate_fn)
    demo.queue()
    demo.launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        prevent_thread_lock=True,
    )

    app = demo.app

    @app.get("/health")
    def health():
        return {"status": "healthy", "model": args.model, "device": str(device)}

    @app.get("/ref_audio/files")
    def list_ref_audio():
        import glob
        files = []
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
            for f in glob.glob(os.path.join(REF_AUDIO_DIR, ext)):
                files.append(os.path.basename(f))
        return {"files": sorted(files)}

    import urllib.parse

    @app.get("/audio/{filename:path}")
    def serve_audio(filename: str):
        from starlette.responses import FileResponse
        safe = urllib.parse.unquote(filename)
        filepath = os.path.join(OUTPUT_DIR, safe)
        if not os.path.exists(filepath):
            return {"error": f"File not found: {safe}"}
        return FileResponse(filepath, media_type="audio/wav")

    @app.post("/api/tts")
    def api_tts(body: dict):
        import gc, soundfile as sf

        text = body.get("text", "").strip()
        if not text:
            return {"error": "text is required"}

        lang = body.get("language")
        if lang and lang == "Auto":
            lang = None

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(body.get("num_step", 32)),
            guidance_scale=float(body.get("guidance_scale", 2.0)),
            denoise=bool(body.get("denoise", True)),
            preprocess_prompt=bool(body.get("preprocess_prompt", True)),
            postprocess_output=bool(body.get("postprocess_output", True)),
        )

        kw = dict(text=text, language=lang, generation_config=gen_config)

        speed = body.get("speed")
        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)

        duration = body.get("duration")
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        mode = body.get("mode", "clone")
        if mode == "clone":
            ref_audio = body.get("ref_audio")
            if not ref_audio:
                return {"error": "ref_audio is required for clone mode"}

            if isinstance(ref_audio, str) and ref_audio.startswith("ref://"):
                filename = ref_audio.removeprefix("ref://")
                ref_audio = os.path.join(REF_AUDIO_DIR, filename)

            if not os.path.exists(ref_audio):
                return {"error": f"File not found: {ref_audio}"}

            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=body.get("ref_text"),
            )
            if hasattr(model, '_asr_pipe') and model._asr_pipe is not None:
                del model._asr_pipe
                model._asr_pipe = None
                torch.cuda.empty_cache()
                gc.collect()

        instruct = body.get("instruct")
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            logger.exception("Generation failed")
            return {"error": f"{type(e).__name__}: {e}"}

        waveform = (audio[0] * 32767).astype("int16")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"omnivoice_{ts}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        sf.write(filepath, waveform, model.sampling_rate)
        logger.info(f"Saved: {filepath}")

        download_url = f"http://127.0.0.1:{args.port}/audio/{urllib.parse.quote(filename)}"

        return {
            "path": filepath,
            "url": download_url,
            "sampling_rate": model.sampling_rate,
            "message": f"Done. Saved: {filepath}",
        }

    import time
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
