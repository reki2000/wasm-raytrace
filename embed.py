import base64
b64 = base64.b64encode(open('dino.wasm', 'rb').read()).decode()
html = open('template.html').read().replace('__WASM_B64__', b64)
open('dino-herd.html', 'w').write(html)
print(f'dino-herd.html: {len(html)} bytes (wasm {len(b64)} b64 chars)')
