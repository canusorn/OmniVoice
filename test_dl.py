import urllib.request, time, sys
url = sys.argv[1] if len(sys.argv) > 1 else "https://huggingface.co/k2-fsa/OmniVoice/resolve/main/model.safetensors"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
start = time.time()
r = urllib.request.urlopen(req, timeout=30)
size = int(r.headers.get("Content-Length", 0))
print(f"File size: {size/1024/1024:.2f} MB")
data = r.read(1024*1024)
elapsed = time.time() - start
print(f"1MB in {elapsed:.2f}s = {1/elapsed:.2f} MB/s")
