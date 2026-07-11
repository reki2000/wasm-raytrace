# DINO // RT HERD — wasm SIMD128 ray-marched dinosaur herd

**🦖 デモページ: <https://reki2000.github.io/wasm-raytrace/>**
(main ブランチへの push で GitHub Actions が自動ビルドして GitHub Pages に公開されます)

依存ゼロ・単一HTMLで動く SDF レイマーチング恐竜デモです。
3種の恐竜(獣脚類 / ステゴサウルス / トリケラトプス)の群れが歩き続けます。
反射する床、フレネル環境反射、プロシージャルテクスチャ入り。GPU 不使用、CPU の
WebAssembly SIMD128 だけでリアルタイムにレイマーチングします。

画面左のパネルから恐竜ごとに表面マテリアル(ノーマル / テクスチャ / 金属反射 /
アクリル半透明)を切り替えられ、反射率・透過率・屈折率などを縦型スライダで
リアルタイムに調整できます。

- emscripten 不要(clang + wasm-ld 直叩き)
- libc 未使用(sin / sqrt / noise は自前実装)
- 出力は `dino-herd.html` 1ファイル(wasm は base64 で埋め込み)

## 必要ツール

- clang + wasm-ld (lld) — wasm32 ターゲット対応版。Ubuntu なら `apt install clang lld`(動作確認は clang 18)
- python3 — base64 埋め込み用
- node — ベンチマーク用(任意)

## ビルド

```sh
./build.sh
```

中身は2ステップ:

```sh
clang --target=wasm32 -msimd128 -mbulk-memory -O3 -ffast-math -fno-math-errno \
  -nostdlib -Wl,--no-entry -Wl,-z,stack-size=65536 \
  -o dino.wasm main.c render.c anim.c dino_model.c
python3 embed.py   # template.html の __WASM_B64__ を置換 → dino-herd.html
```

## ベンチマーク・目視確認

```sh
node test.js   # 480x270 で ms/frame 計測 + f0〜f5.ppm をダンプ(f5 はマテリアル適用例)
```

PPM→PNG は Pillow で `python3 -c "from PIL import Image; Image.open('f0.ppm').save('f0.png')"` など。

## ファイル構成

C ソースは「レンダリングエンジン」「アニメーションエンジン」「恐竜モデル」「統合メイン」の4つに分割されています。

| ファイル | 内容 |
|---|---|
| `main.c` | 統合メイン。エクスポート `render(t,az,el,dist,w,h)` / `fb()` / `mat(i,mode,refl,tran,ior,tex,gloss)` の3本と、各エンジンの呼び出し順 |
| `render.c` / `render.h` | レンダリングエンジン(SDF プリミティブ / カリング / マーチング / 影 / 環境 / マテリアル / フレームパイプライン) |
| `anim.c` / `anim.h` | アニメーションエンジン(クロック / ノースリップ歩行 / 群れ運動) |
| `dino_model.c` / `dino_model.h` | 恐竜3種のジオメトリとアニメーションパラメータ、種別テクスチャ |
| `vec.h` | 共通 SIMD ヘルパーと libm 代替の数学関数 |
| `template.html` | UI・適応解像度・入力処理。`__WASM_B64__` が wasm 埋め込み位置 |
| `embed.py` | wasm を base64 で HTML に焼き込み |
| `build.sh` | 上記2ステップのビルドスクリプト |
| `test.js` | node ベンチ + フレームダンプ |
| `.github/workflows/pages.yml` | main への push で自動ビルド → GitHub Pages 公開 |
| `ARCHITECTURE.md` | コード解説(設計理念・モジュール構成・機能ごとの実装箇所) |

## コード解説

設計理念、モジュール構成、機能ごとの実装箇所・調整ポイントは
**[ARCHITECTURE.md](ARCHITECTURE.md)** にまとめています。改修の際はそちらを入口にしてください。

## GitHub Pages への公開

`.github/workflows/pages.yml` が main ブランチへの push をトリガに、CI 上で
`build.sh` を実行して `dino-herd.html` を `index.html` として GitHub Pages に
デプロイします。初回のみリポジトリの Settings → Pages で
Source を **GitHub Actions** にしてください(workflow 側でも自動有効化を試みます)。

公開 URL: <https://reki2000.github.io/wasm-raytrace/>

## ライセンス

MIT License(`LICENSE` 参照)
