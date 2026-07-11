# DINO // RT HERD — wasm SIMD128 ray-marched dinosaur herd

**🦖 デモページ: <https://reki2000.github.io/wasm-raytrace/>**
(main ブランチへの push で GitHub Actions が自動ビルドして GitHub Pages に公開されます)

依存ゼロで動く SDF レイマーチング恐竜デモです。
3種の恐竜(獣脚類 / ステゴサウルス / トリケラトプス)の群れが歩き続けます。
反射する床、フレネル環境反射、プロシージャルテクスチャ入り。GPU 不使用、CPU の
WebAssembly SIMD128 だけでリアルタイムにレイマーチングします。

画面左のパネルから恐竜ごとに表面マテリアル(ノーマル / テクスチャ / 金属反射 /
アクリル半透明)を切り替えられ、反射率・透過率・屈折率などを縦型スライダで
リアルタイムに調整できます。

## Quaternius 恐竜モデル(三角形メッシュ経路)

左パネルの **MODEL** で [Quaternius Animated Dinosaur Pack](https://quaternius.com/packs/animateddinosaurs.html)
の6体(T-Rex / Velociraptor / Triceratops / Stegosaurus / Parasaurolophus /
Apatosaurus)を選ぶと、**6体を横一列に並べて三角形レイトレ**で描画します。
MODEL ボタンは**カメラの中心をその恐竜へ切り替え**る役割で、中心にした恐竜に対して:

- **ACTION** — Idle / Walk / Run / Attack / Jump / Death を**リアルタイム**切替
- **マテリアル** — SDF 版と同じ4モード(ノーマル / テクスチャ / 金属反射 / アクリル)を
  スライダ(反射率 / 光沢 / 透過率 / 屈折率 / 模様)で調整。個体ごとに保持される

**SDF HERD** ボタンで元の SDF 群れに戻ります。

- SDF 経路とは別の第2レンダラ(`mesh.c`)。同じチェッカー床・空・反射・フォグ・マテリアルで描くので見た目が揃う
- GLB は実行時に `models/*.glb` を fetch。JS(`glb.js`)が glTF スキンアニメをサンプリングして毎フレーム
  ボーン行列を wasm に渡し、wasm 側で全6体を LBS スキニング → 1つの BVH に再構築 → スカラーレイトレ
  (プライマリ + 太陽シャドウ + 床ミラー + アクリル透過)
- ローポリ(6体計 約9500三角形)なので CPU でも実用速度。法線はスキン後の面法線でフラットシェード
- モデルの色は glTF の `baseColorFactor`(画像テクスチャ・UV なし)

Quaternius モデルは **CC0 1.0(パブリックドメイン)**で個人・商用利用可。`models/` に同梱しています(→ [クレジット / ライセンス](#クレジット--ライセンス))。

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
  -o dino.wasm main.c render.c anim.c dino_model.c mesh.c
python3 embed.py   # template.html の __WASM_B64__ を置換 → dino-herd.html
```

wasm は `dino-herd.html` に埋め込みますが、Quaternius モデル機能は実行時に
`glb.js` と `models/*.glb` を同一オリジンで読みます。ローカル確認時は
`python3 -m http.server` などで配信してください(`file://` だと fetch が失敗します)。

## ベンチマーク・目視確認

```sh
node test.js   # 480x270 で ms/frame 計測 + f0〜f5.ppm をダンプ(f5 はマテリアル適用例)
```

PPM→PNG は Pillow で `python3 -c "from PIL import Image; Image.open('f0.ppm').save('f0.png')"` など。

## ファイル構成

C ソースは「レンダリングエンジン」「アニメーションエンジン」「恐竜モデル」「統合メイン」の4つに分割されています。

| ファイル | 内容 |
|---|---|
| `main.c` | 統合メイン。SDF 経路 `render(...)` とメッシュ経路 `renderMesh(...)` / `mesh*` アップロード用エクスポート、`fb()` / `mat(...)` |
| `render.c` / `render.h` | SDF レンダリングエンジン(SDF プリミティブ / カリング / マーチング / 影 / 環境 / マテリアル / フレームパイプライン) |
| `mesh.c` / `mesh.h` | メッシュレンダリングエンジン(LBS スキニング / BVH / 三角形レイトレ / フラットシェード)。Quaternius glTF 用 |
| `glb.js` | GLB パーサ + glTF スケルタルアニメのサンプラ(Node/ブラウザ共用) |
| `anim.c` / `anim.h` | アニメーションエンジン(クロック / ノースリップ歩行 / 群れ運動) |
| `dino_model.c` / `dino_model.h` | 恐竜3種のジオメトリとアニメーションパラメータ、種別テクスチャ |
| `vec.h` | 共通 SIMD ヘルパーと libm 代替の数学関数 |
| `models/*.glb` | Quaternius Animated Dinosaur Pack(6体、実行時 fetch) |
| `template.html` | UI・適応解像度・入力処理。`__WASM_B64__` が wasm 埋め込み位置 |
| `embed.py` | wasm を base64 で HTML に焼き込み |
| `build.sh` | 上記2ステップのビルドスクリプト |
| `test.js` | node ベンチ + フレームダンプ(SDF 経路) |
| `test_mesh.js` | node でメッシュ経路を検証(GLB 読込 → スキン → 描画 → PPM) |
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

## クレジット / ライセンス

- **コード**: MIT License(`LICENSE` 参照)
- **恐竜モデル(`models/*.glb`)**: [Quaternius](https://quaternius.com/) — *Animated Dinosaur Pack*。
  **CC0 1.0 Universal(パブリックドメイン)**。個人・商用利用、改変、再配布いずれも自由で、
  帰属表示は義務ではありません(本プロジェクトは敬意として明記しています)。
  配布元: <https://quaternius.com/packs/animateddinosaurs.html>

モデルのクレジットは実行画面右上にも常時表示されます。
