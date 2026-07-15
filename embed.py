import sys, base64

mt_mem_bytes = int(sys.argv[1]) if len(sys.argv) > 1 else 8388608

b64 = base64.b64encode(open('dino.wasm', 'rb').read()).decode()
b64mt = base64.b64encode(open('dino-mt.wasm', 'rb').read()).decode()
html = open('template.html').read()
html = html.replace('__WASM_B64__', b64)
html = html.replace('__WASM_MT_B64__', b64mt)
html = html.replace('__WASM_MT_PAGES__', str(mt_mem_bytes // 65536))
open('dino-herd.html', 'w').write(html)
print(f'dino-herd.html: {len(html)} bytes (wasm {len(b64)} + mt {len(b64mt)} b64 chars)')
