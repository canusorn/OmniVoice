# OmniVoice — Docker Setup

[OmniVoice](https://github.com/k2-fsa/OmniVoice) by **K2-FSA** (Xiaomi AI Lab Next-gen Kaldi team) — state-of-the-art text-to-speech for **600+ languages** รองรับ **Voice Clone** และ **Voice Design**

## Requirements

- Docker 29+ (ติดตั้งแล้ว)
- NVIDIA GPU + CUDA driver 13.1+
- NVIDIA Container Toolkit (`nvidia-docker`)
- พื้นที่ว่าง ~25 GB สำหรับ Docker image + model cache

## Quick Start

```bash
# Build image (ครั้งแรก ~10-20 นาที)
docker compose build

# Start container
docker compose up -d

# Open browser
start http://localhost:8001
```

container จะดาวน์โหลด model weights ครั้งแรกที่ run (OmniVoice ~6 GB, Whisper ~500 MB) ใช้เวลาตามความเร็วเน็ต แล้ว cache ไว้ถาวรที่ `.\cache\huggingface`

### คำสั่ง

| คำสั่ง | คำอธิบาย |
|--------|----------|
| `docker compose up -d` | เริ่ม container |
| `docker compose down` | หยุด container |
| `docker compose restart` | restart container |
| `docker compose build --no-cache` | rebuild image |
| `docker logs -f omnivoice` | ดู logs |
| `.\status.ps1` | เช็คสถานะ |

## Directory Structure

```
D:\tts\OmniVoice\
├── Dockerfile              # CUDA 12.8 + PyTorch + omnivoice
├── docker-compose.yml      # GPU, port 8001, environment
├── app.py                  # Gradio Web UI + API
├── .env                    # environment variables
├── entrypoint.ps1          # สร้าง directories + start container
├── stop.ps1                # stop container
├── status.ps1              # check status script
├── cache\
│   └── huggingface\        # **persistent** model weights (mount)
├── output\                 # **persistent** generated audio
└── ref_audio\              # **persistent** reference audio files
```

### Bind Mounts

| Host Path | Container Path | คำอธิบาย |
|-----------|---------------|----------|
| `.\cache\huggingface` | `/root/.cache/huggingface` | เก็บ model weights (ดาวน์โหลดครั้งเดียว) |
| `.\output` | `/output` | ไฟล์เสียงที่ generate จะเซฟอัตโนมัติ |
| `.\ref_audio` | `/ref_audio` | วางไฟล์ audio สำหรับ voice cloning |
| `.\app.py` | `/app/app.py` | แก้ไข app.py โดยไม่ต้อง rebuild |

## Usage — Web UI

เปิด `http://localhost:8001` จะเจอ 2 tabs:

### Voice Clone
1. เลือก tab **Voice Clone**
2. พิมพ์ข้อความใน **Text**
3. เลือก **Reference Audio** (ไฟล์ .wav/.mp3 ของเสียงที่ต้องการ clone)
4. กด **Generate**
5. ไฟล์ .wav จะ auto-save ที่ `.\output\`

### Voice Design
1. เลือก tab **Voice Design**
2. พิมพ์ข้อความใน **Text**
3. ปรับ **Speaker Attributes** (pitch, speed, emotion ฯลฯ)
4. กด **Generate**
5. ไฟล์ .wav จะ auto-save ที่ `.\output\`

## API Usage

### `/api/tts` — Simple JSON API (recommended)

ไม่ต้อง upload file, ส่ง `ref://filename.wav` อ้างอิงไฟล์ใน `ref_audio/` โดยตรง

```bash
curl -X POST http://localhost:8001/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "สวัสดีครับ",
    "language": "Thai",
    "ref_audio": "ref://my_voice.wav",
    "mode": "clone",
    "num_step": 8,
    "guidance_scale": 4.0
  }'
```

```python
import requests

r = requests.post("http://localhost:8001/api/tts", json={
    "text": "Hello world",
    "language": "English",
    "ref_audio": "ref://my_voice.wav",
    "mode": "clone",        # "clone" or "design"
    "num_step": 8,
    "guidance_scale": 4.0,
})
print(r.json())  # returns audio + message
```

### `/gradio_api/` — Raw Gradio 6 API (low-level)

ใช้ส่ง file โดยตรง + event polling:

```python
import requests, time

API = "http://localhost:8001/gradio_api"

# Upload reference audio first
r = requests.post(f"{API}/upload", files={"files": open("my_voice.wav", "rb")})
path = r.json()[0]

# Send request → get event_id
r = requests.post(f"{API}/call/_clone_fn", json={
    "data": [
        "สวัสดีครับ", "Thai",
        {"path": path, "meta": {"_type": "gradio.FileData"}},  # ref_audio
        None,     # ref_text
        None,     # instruct
        8,        # num_step
        4.0,      # guidance_scale
        True,     # denoise
        1.0,      # speed
        None,     # duration
        False,    # preprocess_prompt
        True,     # postprocess_output
    ]
})
event_id = r.json()["event_id"]

# Poll for result
for _ in range(60):
    time.sleep(2)
    r = requests.get(f"{API}/call/_clone_fn/{event_id}")
    if "event: complete" in r.text:
        print("Done!")
        break
```

### Function Endpoints

| Function | Endpoint | Parameters (data array) |
|----------|----------|------------------------|
| Voice Clone | `/gradio_api/call/_clone_fn` | `[text, lang, ref_audio, ref_text, instruct, ns, gs, dn, sp, du, pp, po]` |
| Voice Design | `/gradio_api/call/_design_fn` | `[text, lang, ns, gs, dn, sp, du, pp, po, ...groups]` |

### List ref_audio Files

```bash
curl http://localhost:8001/ref_audio/files
```
```python
requests.get("http://localhost:8001/ref_audio/files").json()
# => {"files": ["my_voice.wav"]}
```

### Language Values

ใช้ชื่อภาษาเต็ม (ไม่ใช่รหัส ISO) เช่น `"Thai"`, `"English"`, `"Auto"` (600+ languages)

### Parameters — Voice Clone (`_clone_fn`)

| Index | Parameter | Type | Default | Description |
|-------|-----------|------|---------|-------------|
| 0 | `text` | string | — | ข้อความ |
| 1 | `language` | string | `"Auto"` | ชื่อภาษาเต็ม |
| 2 | `ref_audio` | object | — | `{"path": "...", "meta": {"_type": "gradio.FileData"}}` |
| 3 | `ref_text` | string\|null | `null` | ข้อความของ ref audio (ถ้าไม่ให้ ASR จะถอดเสียงให้) |
| 4 | `instruct` | string\|null | `null` | instruction prompt |
| 5 | `num_step` | int | `32` | diffusion steps |
| 6 | `guidance_scale` | float | `4.0` | guidance scale (min 4) |
| 7 | `denoise` | bool | `True` | denoising |
| 8 | `speed` | float | `1.0` | ความเร็ว |
| 9 | `duration` | float\|null | `null` | ความยาว (วินาที) |
| 10 | `preprocess_prompt` | bool | `False` | preprocess |
| 11 | `postprocess_output` | bool | `True` | postprocess |

### Parameters — Voice Design (`_design_fn`)

| Index | Parameter | Type | Default | Description |
|-------|-----------|------|---------|-------------|
| 0 | `text` | string | — | ข้อความ |
| 1 | `language` | string | `"Auto"` | ชื่อภาษาเต็ม |
| 2 | `num_step` | int | `32` | diffusion steps |
| 3 | `guidance_scale` | float | `4.0` | guidance scale (min 4) |
| 4 | `denoise` | bool | `True` | denoising |
| 5 | `speed` | float | `1.0` | ความเร็ว |
| 6 | `duration` | float\|null | `null` | ความยาว (วินาที) |
| 7 | `preprocess_prompt` | bool | `False` | preprocess |
| 8 | `postprocess_output` | bool | `True` | postprocess |
| 9+ | `groups` | dropdowns | — | speaker attributes |

## Configuration

### Environment Variables (docker-compose.yml)

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | `/output` | output directory |
| `REF_AUDIO_DIR` | `/ref_audio` | reference audio directory |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace cache root |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `3600` | download timeout (seconds) |
| `HF_HUB_ENABLE_XET` | `0` | disable XET protocol |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | suppress progress bars |

### app.py CLI flags

```bash
# รันด้วย ASR model
docker compose exec omnivoice python3 /app/app.py --load-asr

# รันด้วย custom port
docker compose exec omnivoice python3 /app/app.py --port 8080

# รันด้วย device CPU
docker compose exec omnivoice python3 /app/app.py --device cpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `k2-fsa/OmniVoice` | HuggingFace model ID |
| `--ip` | `0.0.0.0` | bind address |
| `--port` | `8001` | server port |
| `--device` | `auto` | device (`cuda`, `cpu`) |
| `--load-asr` | `False` | โหลด ASR model (Whisper) |
| `--asr-model` | `openai/whisper-large-v3-turbo` | ASR model name |
| `--share` | `False` | สร้าง public Gradio link |

## Troubleshooting

### Model download ตัด / timeout
```bash
# เช็คสถานะ download
docker exec omnivoice ls -la /root/.cache/huggingface/hub/models--k2-fsa--OmniVoice/blobs/

# ลบ cache แล้วลองใหม่
docker compose down
Remove-Item -Recurse -Force .\cache\huggingface -ErrorAction SilentlyContinue
docker compose up -d
```

### GPU ไม่ถูก detect
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```

### VRAM ไม่พอ
- ลด `num_step` (16 แทน 32)
- ใช้ `dtype=torch.float16` (default)
- ไม่ load ASR model (ถ้าไม่จำเป็น)
- ปิด browser tabs อื่นที่ใช้ GPU
