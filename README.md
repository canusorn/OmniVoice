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

Gradio API can be accessed at `http://localhost:8001/gradio_api/`

### Get API Info

```bash
curl -X POST http://localhost:8001/gradio_api/api/info
```

### Predict (call a function)

```bash
curl -X POST http://localhost:8001/gradio_api/call/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [...]
  }'
```

### Python Example

```python
import requests

API_URL = "http://localhost:8001/gradio_api/call/predict"

# Voice synthesis
response = requests.post(API_URL, json={
    "data": [
        "สวัสดีครับ ยินดีต้อนรับ",   # text
        "th",                        # language
        None,                        # ref_audio (filepath/None)
        None,                        # instruct
        32,                          # num_step
        2.0,                         # guidance_scale
        True,                        # denoise
        1.0,                         # speed
        None,                        # duration
        False,                       # preprocess_prompt
        True,                        # postprocess_output
        "tts"                        # mode: "tts" | "clone"
    ]
})

print(response.json())
```

### Voice Clone via API

```python
import requests

# Upload reference audio first
with open("speaker.wav", "rb") as f:
    upload = requests.post(
        "http://localhost:8001/upload",
        files={"files": f}
    )
    ref_path = upload.json()[0]

# Clone voice
response = requests.post(
    "http://localhost:8001/gradio_api/call/predict",
    json={
        "data": [
            "Hello, this is a voice clone.",
            "en",
            ref_path,
            None,
            32,
            2.0,
            True,
            1.0,
            None,
            False,
            True,
            "clone"
        ]
    }
)

result = response.json()
print(result)
```

### JavaScript Example

```javascript
const API_URL = "http://localhost:8001/gradio_api/call/predict";

async function generateTTS(text, mode = "tts") {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: [text, "th", null, null, 32, 2.0, true, 1.0, null, false, true, mode]
    })
  });
  const result = await res.json();
  return result;
}
```

### Parameters (data array order)

| Index | Parameter | Type | Default | Description |
|-------|-----------|------|---------|-------------|
| 0 | `text` | string | — | ข้อความที่จะสังเคราะห์เสียง |
| 1 | `language` | string | `"Auto"` | ภาษา (`"th"`, `"en"`, `"zh"`, `"ja"` หรือ `"Auto"`) |
| 2 | `ref_audio` | string\|null | `null` | path reference audio (เฉพาะ mode "clone") |
| 3 | `instruct` | string\|null | `null` | instruction prompt |
| 4 | `num_step` | int | `32` | number of diffusion steps |
| 5 | `guidance_scale` | float | `2.0` | classifier-free guidance scale |
| 6 | `denoise` | bool | `True` | ใช้ denoising |
| 7 | `speed` | float | `1.0` | ความเร็ว (1.0 = ปกติ) |
| 8 | `duration` | float\|null | `null` | กำหนดความยาว (วินาที) |
| 9 | `preprocess_prompt` | bool | `False` | preprocess prompt text |
| 10 | `postprocess_output` | bool | `True` | postprocess output audio |
| 11 | `mode` | string | `"tts"` | `"tts"` หรือ `"clone"` |

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
| `--asr-model` | `openai/whisper-tiny` | ASR model name |
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
