# DINO // RT HERD — wasm SIMD128 ray-marched dinosaur herd

依存ゼロで動く、WebAssembly SIMD128 によるリアルタイム SDF レイマーチング恐竜デモです。
GPU を使わず、CPU の wasm だけで恐竜の群れを描画します。Quaternius 製の三角形メッシュ
恐竜モデル(6種)の描画にも対応しています。

- emscripten 不要(clang + wasm-ld 直叩き)、libc 未使用
- 出力は `dino-herd.html` 1ファイル(wasm は base64 で埋め込み)

## デモ

🦖 <https://wasm-raytrace.pages.dev/>

## 操作方法

- ドラッグ: カメラの周回(オービット) / ホイール・ピンチ: ズーム
- 左パネル: 恐竜ごとにマテリアル(ノーマル / テクスチャ / 金属反射 / アクリル半透明)と
  反射率・透過率・屈折率などをリアルタイムに調整
- **MODEL** ボタン: Quaternius メッシュ恐竜(T-Rex / Velociraptor / Triceratops /
  Stegosaurus / Parasaurolophus / Apatosaurus)に切り替え、ACTION(Idle/Walk/Run/
  Attack/Jump/Death)を選択可能
- **SDF HERD** ボタン: 元の SDF 群れ表示に戻る

## 必要ツール

- clang + wasm-ld (lld) — wasm32 ターゲット対応版。Ubuntu なら `apt install clang lld`(動作確認は clang 18)
- python3 — base64 埋め込み用

## ビルド

```sh
./build.sh
```

`dino-herd.html` が生成されます。ブラウザで直接開けますが、Quaternius モデル機能は
実行時に `glb.js` と `models/*.glb` を同一オリジンで fetch するため、ローカル確認時は
`python3 -m http.server` などで配信してください(`file://` だと fetch が失敗します)。

## ファイル構成・コード解説

各ファイルの役割、設計理念、機能ごとの実装箇所は **[ARCHITECTURE.md](ARCHITECTURE.md)**
にまとめています。改修の際はそちらを入口にしてください。

## デプロイ

main ブランチへの push で `.github/workflows/cloudflare.yml` が自動ビルドして
Cloudflare Pages に公開します。

## クレジット / ライセンス

- **コード**: MIT License(`LICENSE` 参照)
- **恐竜モデル(`models/*.glb`)**: [Quaternius](https://quaternius.com/) — *Animated Dinosaur Pack*。
  **CC0 1.0 Universal(パブリックドメイン)**。個人・商用利用、改変、再配布いずれも自由で、
  帰属表示は義務ではありません(本プロジェクトは敬意として明記しています)。
  配布元: <https://quaternius.com/packs/animateddinosaurs.html>

モデルのクレジットは実行画面右上にも常時表示されます。
