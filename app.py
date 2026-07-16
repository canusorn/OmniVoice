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

    @app.post("/api/tts")
    def api_tts(body: dict):
        import time as _time, uuid, json as _json, httpx

        mode = body.get("mode", "clone")
        ref_audio = body.get("ref_audio")

        with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=600) as client:

            if isinstance(ref_audio, str) and ref_audio.startswith("ref://"):
                filename = ref_audio.removeprefix("ref://")
                src = os.path.join(REF_AUDIO_DIR, filename)
                if not os.path.exists(src):
                    return {"error": f"File not found: {filename}"}
                with open(src, "rb") as f:
                    upload = client.post("/gradio_api/upload", files={"files": ("file", f, "audio/wav")})
                    upload.raise_for_status()
                    ref_audio = {"path": upload.json()[0], "meta": {"_type": "gradio.FileData"}}

            if mode == "clone":
                fn = "_clone_fn"
                data = [
                    body.get("text", ""),
                    body.get("language", "Auto"),
                    ref_audio,
                    body.get("ref_text"),
                    body.get("instruct"),
                    body.get("num_step", 32),
                    body.get("guidance_scale", 4.0),
                    body.get("denoise", True),
                    body.get("speed", 1.0),
                    body.get("duration"),
                    body.get("preprocess_prompt", False),
                    body.get("postprocess_output", True),
                ]
            else:
                fn = "_design_fn"
                data = [
                    body.get("text", ""),
                    body.get("language", "Auto"),
                    body.get("num_step", 32),
                    body.get("guidance_scale", 4.0),
                    body.get("denoise", True),
                    body.get("speed", 1.0),
                    body.get("duration"),
                    body.get("preprocess_prompt", False),
                    body.get("postprocess_output", True),
                ]

            resp = client.post(f"/gradio_api/call/{fn}", json={"data": data})
            resp.raise_for_status()
            event_id = resp.json()["event_id"]

            for _ in range(150):
                _time.sleep(2)
                resp = client.get(f"/gradio_api/call/{fn}/{event_id}")
                text = resp.text
                if '"error"' in text and '"error": null' not in text:
                    return {"error": text}
                if "event: complete" in text:
                    for line in text.split("\n"):
                        if line.startswith("data: "):
                            return _json.loads(line.removeprefix("data: "))
                    return {"result": text}

            return {"error": "timeout"}

    import time
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
