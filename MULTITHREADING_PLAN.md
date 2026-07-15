# マルチスレッド化計画 — wasm threads によるレンダリング並列化

emscripten を導入せず、現行の clang + wasm-ld 直叩き・依存ゼロ・1ファイル配布を維持したまま、
wasm threads(shared memory + atomics)+ Web Worker プールでピクセルループを並列化する。

## 現状分析(なぜ行並列が刺さるか)

- 描画経路は2つ: SDF 群れ(`render.c renderFrame`)と三角形メッシュ(`mesh.c renderMesh`)。
  どちらも `for (int y=0; y<h; y++)` の行ループがフレーム時間のほぼ全部。
- フレーム毎の可変状態(`PR`/`DB`/`EYP`/`BONE`/`SV`/BVH ノード/`packTris` の SoA …)は
  **すべて描画前フェーズで書き込まれ、ピクセルループ中は読み取り専用**。
  ピクセルは互いに独立・決定的で、`FB` への書き込みは行ごとに素(disjoint)。
  → 行単位の並列化にデータ競合が一切ない理想的な構造。
- 移設が必要な唯一の例外: `renderMesh` 内の `static int frameNo, lastNT`(BVH 再構築の
  周期管理)。準備フェーズ関数に移す。
- ベースライン(4コアコンテナ、480×270): SDF 36.2 ms/frame、mesh 49.0 ms/frame。
- 配信は Cloudflare Pages(`.github/workflows/cloudflare.yml`)。`_headers` ファイルで
  COOP/COEP を配信でき、SharedArrayBuffer の要件(crossOriginIsolated)を満たせる。

## 全体方針

- **wasm threads**: 全インスタンスが共有する `WebAssembly.Memory({shared:true})` 上で、
  ワーカー N 本 + メインスレッドが**行ワークスチール**でピクセルを分担する。
- **pthread 実装は作らない**。必要なのは「共有メモリ + アトミックな行カウンタ +
  フレーム開始/完了の同期」だけなので、自前の最小基盤で足りる。
- **非対応環境フォールバック**: shared-memory wasm は非分離環境では instantiate できない
  ため、シングルスレッド版バイナリも併せて埋め込み、実行時に選択する。

## Phase 0 — ビルド / ランタイム基盤

1. **ビルドフラグ**(MT 版):
   - コンパイル: 既存フラグ + `-matomics`(`-mbulk-memory` は既にある。
     `--shared-memory` は全 TU に atomics+bulk-memory を要求)
   - リンク: `-Wl,--shared-memory,--import-memory,--max-memory=67108864,--export=__stack_pointer`
   - shared memory ではデータセグメントが passive になり、wasm-ld が atomic ガード付きの
     `__wasm_init_memory` を生成する。多重 instantiate でもメモリ初期化は1回だけ走る。
2. **2バイナリ出力**: `build.sh` で `dino.wasm`(現行 ST 版)と `dino-mt.wasm` を両方
   ビルドし、`embed.py` を拡張して両方 base64 埋め込み(現行 wasm は 77KB なので
   HTML は +100KB 程度。gzip 後の増分はさらに小さい)。
3. **メモリ import 化**(MT 版のみ): JS が shared Memory を作って全インスタンスに渡す。
   必要 initial ページ数はリンカが wasm の import エントリに埋めるので、`embed.py` が
   それをパースして HTML に定数として出力する(手書きの数字ずれを防ぐ)。
4. **ワーカースタック**: C 側に `static char TSTACK[MAXTHREADS][STACK_SZ]`(64KB×8 程度)を
   確保し先頭アドレスを export。各ワーカーは instantiate 直後に
   `exports.__stack_pointer.value = base + (id+1)*STACK_SZ`(16B アライン、下方伸長)を設定。
   TLS は現状使っていないので導入しない。
5. **ワーカープール**: template.html にワーカースクリプトを Blob URL で内包(1ファイル
   配布を維持)。メインで `WebAssembly.compile` した Module + shared Memory + worker id を
   postMessage で配布し、各ワーカーが instantiate。プールは常駐(起動コストは初回のみ)。
   プールサイズは `min(navigator.hardwareConcurrency - 1, 7)`、メインも計算に参加。
6. **実行時選択**: `crossOriginIsolated && typeof SharedArrayBuffer === 'function'` なら
   MT 版、でなければ現行どおり ST 版をメインスレッドで instantiate。以降のフレーム
   ループは共通コードで、MT 時だけオーケストレーションが変わる。

## Phase 1 — ピクセルループの並列化(本丸)

1. **C 側の分割**(両経路とも「準備」と「行レンダリング」に分ける):
   - SDF: `render(t,…)` → `renderPrep(t)`(`animTick`+`animate`。メインインスタンス専用)
     + `renderRows(az,el,dist,w,h)`(ワーカー/メイン共用)。カメラ導出は引数からの
     純関数なので各スレッドがローカルに再計算する(共有 struct 不要で最小差分)。
   - mesh: `meshPrep(...)`(`skin` + `buildBVH`/`refitBVH` + `packTris` + SUN。
     `frameNo`/`lastNT` もここへ移動)+ `renderMeshRows(az,el,dist,w,h)`。
2. **動的ワークスチール**: 共有の `_Atomic int nextRow` に対し
   `__c11_atomic_fetch_add` で **4〜8行単位のチャンク**を取得するループを
   `renderRows` 系の内部に持つ。恐竜が画面中央に集中して行コストが不均一なので、
   静的な帯分割より効く。エクスポートは `frameBegin(w,h)`(カウンタリセット)と
   `renderRows*(…)` の2種で足りる。
3. **フレームオーケストレーション**(JS、rAF 1回分):
   1. (mesh 時)ボーン行列を共有メモリへ書き込み
   2. メインインスタンスで `renderPrep` / `meshPrep` を実行(逐次)
   3. `frameBegin` で行カウンタをリセット、共有 Int32Array の世代カウンタを進めて
      `Atomics.notify` → ワーカーは `Atomics.wait` ループから起きて行スチール開始
   4. **メインスレッドも `renderRows` に参加**(待ち時間を計算に充てる)
   5. 完了同期: done カウンタを各スレッドが inc。メインは自分の分を終えたあと
      `Atomics.waitAsync` で残りを待つ(2026 年時点で全主要ブラウザ対応。
      未対応環境は postMessage 通知にフォールバック)
   6. 揃ったら blit して次フレームへ
4. **blit の修正**: ImageData は SharedArrayBuffer ビューを受け付けないため、MT 時は
   FB をレベル別の非共有 `Uint8ClampedArray` に1回コピーしてから `putImageData`
   (1280×720 で 3.6MB の memcpy、1ms 未満)。ST 時は現行のゼロコピーのまま。

**期待値**: ピクセルフェーズはスレッド数にほぼ線形。4コアで両経路 3〜3.5×
(mesh は skin+refit+packTris の逐次分が Amdahl で残る)。適応解像度
(`LV`/`ema` の 15ms/7.5ms しきい値)はそのままで、自然に上位解像度へ張り付く。

## Phase 2 — 準備フェーズの並列化(計測してから)

- Phase 1 完了後に `meshPrep` の内訳(skin / refit / packTris)を計測し、逐次部が
  フレーム時間の 15% を超えるようなら着手:
  - `skin()`(19k 頂点)と `packTris()`(9.5k 三角形)は行スチールと同じ
    アトミックカウンタで範囲並列化できる(書き込み先が index ごとに素)。
  - `buildBVH` は 16 フレームに1回なので優先度低。`refitBVH` は逐次でも軽い想定。
- SDF 側の `animate()` は軽量なので対象外。

## Phase 3 — 配信・互換

- リポジトリに `_headers` を追加し、`cloudflare.yml` で `_site/` へコピー:

  ```
  /*
    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp
  ```

- アセット(`models/*.glb`, `glb.js`)はすべて同一オリジンなので COEP の影響なし。
- 非分離環境(file:// 直開き、ヘッダを付けられないホスティング)では ST 版に
  自動フォールバックし、現行と同じ挙動を保証。
- HUD に `MT×N` / `ST` のスレッド表示を追加(動作確認と実測の見える化)。

## Phase 4 — 検証・ベンチマーク

- **正当性**: ピクセルは独立・決定的なので、1スレッドと N スレッドで出力は
  バイト一致するはず。test.js の PPM ダンプを ST/MT で比較して確認する。
- **ヘッドレスベンチ**: `test_mt.js` を追加。node の `worker_threads` は shared wasm
  memory に対応しているので、ブラウザなしでスケーリングを計測できる。
  両経路 × スレッド数 {1,2,4,8} の ms/frame を出力。
- **ブラウザ実測**: Chrome / Firefox / Safari(iOS 含む)で HUD の ms と解像度レベルを確認。

## リスク・注意点

| 項目 | 内容 | 対応 |
|---|---|---|
| SAB 要件 | crossOriginIsolated でないと MT 版が動かない | `_headers` + ST フォールバック |
| ImageData | SAB ビュー不可 | MT 時のみ非共有バッファへコピー |
| メインの Atomics.wait 禁止 | メインスレッドはブロック待ち不可 | 参加型スチール + `Atomics.waitAsync` |
| スタック衝突 | 全インスタンスがリンカ既定の同一スタック領域を指す | `TSTACK` + `__stack_pointer` export で分離 |
| HTML サイズ | 2バイナリ埋め込みで +100KB 程度 | 許容(gzip 後は僅差)。気になれば ST 版を後日削除 |
| 準備フェーズの逐次残り | mesh 経路の Amdahl 上限 | Phase 2 で計測して判断 |

## PR 分割案

1. **PR1**: Phase 0 + SDF 経路の行並列 + `_headers` + フォールバック
   (デモが Cloudflare 配信なので COOP/COEP は初回 PR に含める)
2. **PR2**: mesh 経路の行並列(`meshPrep` 分割含む)+ `test_mt.js` + HUD 表示
3. **PR3**: (計測結果次第)`skin`/`packTris` の並列化
