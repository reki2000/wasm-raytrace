# DINO // RT HERD — wasm SIMD128 ray-marched dinosaur herd

**🦖 デモページ: <https://reki2000.github.io/wasm-raytrace/>**
(main ブランチへの push で GitHub Actions が自動ビルドして GitHub Pages に公開されます)

依存ゼロ・単一HTMLで動く SDF レイマーチング恐竜デモです。
3種の恐竜(獣脚類 / ステゴサウルス / トリケラトプス)の群れが歩き続けます。
反射する床、フレネル環境反射、プロシージャルテクスチャ入り。GPU 不使用、CPU の
WebAssembly SIMD128 だけでリアルタイムにレイマーチングします。

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
  -o dino.wasm dino.c
python3 embed.py   # template.html の __WASM_B64__ を置換 → dino-herd.html
```

## ベンチマーク・目視確認

```sh
node test.js   # 480x270 で ms/frame 計測 + f0〜f4.ppm をダンプ
```

PPM→PNG は Pillow で `python3 -c "from PIL import Image; Image.open('f0.ppm').save('f0.png')"` など。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `dino.c` | 全部入り(SIMDヘルパー / SDF / 群れ挙動 / 3種ジオメトリ / レンダラ)。エクスポートは `render(t,az,el,dist,w,h)` と `fb()` |
| `template.html` | UI・適応解像度・入力処理。`__WASM_B64__` が wasm 埋め込み位置 |
| `embed.py` | wasm を base64 で HTML に焼き込み |
| `build.sh` | 上記2ステップのビルドスクリプト |
| `test.js` | node ベンチ + フレームダンプ |
| `.github/workflows/pages.yml` | main への push で自動ビルド → GitHub Pages 公開 |

## 設計理念

1. **単一HTML・依存ゼロ** — 配布物はブラウザで開くだけの1ファイル。ビルドも clang / python3 のみで、emscripten のランタイムグルーも npm も使わない。wasm は `-nostdlib` のフリースタンディングで、数学関数(`fsin` / `vsin` / `fsqrt`)まで自前実装している。
2. **決定論的・ステートレスなアニメーション** — 群れの位置・歩行位相・咀嚼などすべてが時刻 `t` の解析関数。フレーム間に状態を持たないので、どの解像度・フレームレートでも同じ絵になり、巻き戻し・ベンチも自由。
3. **SIMD は SoA(Structure of Arrays)で4ピクセル同時** — v128 の4レーン = 横並び4ピクセル。ベクトル `V3` は「x4本, y4本, z4本」の SoA 持ちで、分岐はレーンマスク(`sel` / `vand`)で表現する。`if (any(mask))` によりパケット全体が不要な処理(反射・影など)を丸ごとスキップできる。
4. **距離関数の Lipschitz 連続性を守る** — 全プリミティブは Lipschitz 定数 ≤ 1 を保証する形でしか変形しない(異方性は「太らせる」方向のみ)。これにより sphere tracing が絶対に面を突き抜けない。
5. **カリングの多段化で O(プリム数) を回避** — 個体ごとの境界球 → レイに近いプリムだけのリスト(`buildList`)→ プリムごとの距離早期棄却、の3段でプリム84個を毎ピクセル評価しない。
6. **物理的に足が滑らない歩行** — 見た目の調整ではなく「接地中の足は地面フレームで厳密に静止する」というケイデンス拘束 K = π·ds/A を数式で守る(後述)。

## 機能要件ごとの実装箇所

改修時の入口として、機能ごとに実装位置・仕組みをまとめます。行番号は `dino.c` / `template.html` のものです。

### 1. WASM ビルドと単一HTML化

| 項目 | 実装箇所 |
|---|---|
| コンパイル(wasm32 + SIMD128) | `build.sh` |
| base64 埋め込み | `embed.py`(`template.html` の `__WASM_B64__` を置換) |
| wasm 初期化・SIMD 非対応検出 | `template.html:56-67`(instantiate 失敗で `#err` 表示) |
| エクスポート API | `dino.c:599-603` — `fb()`(フレームバッファ先頭ポインタ)と `render(t, az, el, dist, w, h)` の2つだけ |

**技術**: JS 側は `new ImageData(new Uint8ClampedArray(e.memory.buffer, fbp, w*h*4), w, h)`(`template.html:72-73`)で wasm 線形メモリを**ゼロコピー**で Canvas に転写する。wasm→JS のピクセルコピーは発生しない。

### 2. フレームループ・適応解像度・入力(`template.html`)

| 機能 | 実装箇所 | 仕組み |
|---|---|---|
| メインループ | `template.html:130-164` `frame()` | rAF 駆動。`render()` の実測 ms を EMA 平滑化 |
| 適応解像度 | `template.html:70-75, 142-145` | 5段階(320×180〜960×540)。EMA>15ms で降格、<7.5ms で昇格。切替は800ms間隔 |
| 表示スケーリング | `template.html:147-153` | オフスクリーン canvas に putImageData → cover-fit で `drawImage`(`image-rendering:pixelated`) |
| カメラ操作 | `template.html:88-121` | drag=orbit(az/el)、wheel/pinch=zoom。4秒無操作で自動オービット(`:134`) |
| HUD | `template.html:123-127, 155-161` | fps / 解像度 / ms を 500ms ごとに更新 |

### 3. SIMD 基盤とデータ構造(`dino.c:10-41`)

| 型 | 定義 | 役割 |
|---|---|---|
| `v4` (= `v128_t`) | `dino.c:10` | f32×4レーン = 4ピクセル分のスカラー値。`S(x)` で splat |
| `V3` | `dino.c:35` | `{v4 x,y,z}` の **SoA 3次元ベクトル**(4本のベクトルを同時に表す) |
| `C3` | `dino.c:36` | `{v4 r,g,b}` の RGB(4ピクセル分の色) |
| レーンマスク分岐 | `dino.c:23` `sel(a,b,m)` | `m ? a : b`(`wasm_v128_bitselect`)。以降すべての「if」はこれで表現 |

スカラー数学(libm 代替)は `dino.c:44-52`: `fsqrt`/`ffabs` は `__builtin_*`、`fsin` は Bhaskara 型の放物線近似(SIMD 版 `vsin` は `dino.c:67-72`)。2D 値ノイズは `dino.c:75-86`(`hash = fract(sin(...)·43758)` の定番構成、2オクターブで使用)。

### 4. SDF プリミティブ: 異方性テーパーカプセル(`dino.c:88-178`)

全ジオメトリ(84プリム)はこの1種類のプリミティブ + 目の球だけで構成されています。

- **データ構造**: `PR[MAXP][16]`(`dino.c:94`)。1プリム = float×16:

  | index | 内容 |
  |---|---|
  | 0-2 | 端点 a |
  | 3-5 | b−a(軸ベクトル) |
  | 6 | 1/&#124;b−a&#124;²(射影の除算回避) |
  | 7, 8 | 半径 r1, r2−r1(テーパー) |
  | 9-11 | カリング用中心 |
  | 12 | 境界半径(SK + 半長 + rmax) |
  | 13-15 | 異方性 Ax, Ay, Az |

- **登録**: `cone()` / `coneA()`(`dino.c:98-117`)。`OX/OZ`(`dino.c:96`)が「今ビルド中の個体」の原点オフセットで、各種ビルダーはローカル座標で書ける。
- **距離評価**: `mapL()`(`dino.c:162-178`)。`d = √(Ax·qx² + Ay·qy² + Az·qz²) − r(h)`。A≤1 でその軸方向に 1/√A 倍「太る」。**Lipschitz 定数は 1 以下のまま**なので sphere tracing 安全。制約: セグメント軸は A=1 の軸に沿わせること。ステゴの背板(z 薄・x 幅広、`dino.c:348`)とトリケラのフリル(x 薄・y/z 幅広、`dino.c:387`)はこれで各1プリム。
- **ブレンド**: `smin()`(`dino.c:125-128`)、滑らかさは `SK = 0.062`(`dino.c:93`。変更時は境界半径計算との整合に注意)。
- **目**: smin に混ぜず**ハード min の球**として別管理(`dino.c:119-142` `EYP/EYRAD/EB`、合成は `mapLE()` `dino.c:180-189`)。輪郭がにじまず、albedo/スペキュラの判定(`eyeDistAll < 0.006`)にも再利用。
- **カリング**: `buildList()`(`dino.c:145-160`)がレイ区間 [tA,tB] にプリム境界球が接近するものだけを u8 リスト化。以降のマーチ・法線・AO はこのリストのみ評価。

### 5. 群れ挙動(`dino.c:191-216, 406-435`)

- **カメラは群れの重心付近に固定**で、世界の方が流れる(地面スクロール `SCROLL = VG·t`、`dino.c:409`)。基準速度 `VG = 0.85`(`dino.c:200`)。
- **各個体の前後ドリフト**は `kin()`(`dino.c:205-216`)が解析的に定義:
  `x(t) = x0 + D·sin(u + a·sin u)`, `u = ωt + φ`(スキュー正弦)。
  実効速度 `speed(t) = VG + dx/dt` が 0.72〜1.19 を往復し「ゆっくり遅れて、たまに駆け足で追いつく」動きになる。広がりは最大 2.0m に有界。
- `Kin` 構造体(`dino.c:204`): `x`(位置)/ `dist`(積分済み移動距離 = 歩行位相の源)/ `speed` / `run`(0..1 の駆け足度。前傾・歩幅増などに使用)。
- **個体の登録**は `animate()`(`dino.c:406-435`): 種ごとにパラメータの違う `kin()` を呼び、`DPR[i]`(プリム範囲)、`DB[i]`(境界球)、`DXW/DZW`(テクスチャ用ワールドオフセット)を記録。太陽方向 `SUNX/Y/Z` もここで設定。

### 6. ノースリップ歩行(`dino.c:54-65` + 各種ビルダー)

`gaitFoot(ph, ds, A, *lift)`:
- 接地区間(位相 < デューティ `ds`)は足を移動距離に対して**線形に後送**、遊脚は smoothstep で前方復帰 + 正弦リフト。
- ケイデンスを **K = π·ds/A**(A = 半歩幅)に拘束すると、接地中の足は地面フレームで厳密に静止する(転がり接触と同じ原理)。
- ガイト位相は積分済み移動距離 `dist·K` から取る(theropod `dino.c:245` の 5.585、stego `:319` の 7.791、trice `:367` の 8.469 がそれぞれの K)ので、**加減速中もスリップしない**。
- ⚠️ 歩幅 `A` や `ds` を変えたら、この係数も π·ds/A に合わせて更新すること。

### 7. 3種のジオメトリ

| 種 | 実装箇所 | 特徴 |
|---|---|---|
| 獣脚類 theropod | `dino.c:218-282`(脚 `tleg` `:219-242`) | 二脚。**2ボーン IK**(`tleg` 内、`dino.c:226-234` で膝位置を余弦定理から解析解)。走行時前傾(`ln`)、顎のヒンジ開閉 `gape`(`:256-259`)、3本指の足 + 2本爪の手 |
| ステゴサウルス stego | `dino.c:302-363` | アーチ体。**背板9枚**は `spineTop()`(`:303-312`、体のカプセル鎖を正確にミラーした背面高さ関数)から基部を取り背骨に 3cm 埋め込み(尾側の板は尾スウェイ追従 `:346`)。尾スパイク4本(thagomizer `:353-356`)。四脚トロット(対角ペア `:358-362`) |
| トリケラトプス trice | `dino.c:365-404` | フリル = 異方性1プリム(`:384-389`)、眉角2 + 鼻角(`:391-393`)、嘴、突進時に頭が下がる(`hd`)。四脚 |

共通: 四足の脚は `qleg()`(`dino.c:284-300`、固定膝曲げ + 蹄トゥ2本)。全種、眼球2(高スペキュラ)+ 下顎(四足2種はゆっくり咀嚼 `chew`)。

### 8. レンダリングパイプライン(`dino.c:602-867` `render()`)

処理順に:

| ステージ | 実装箇所 | 内容 |
|---|---|---|
| カメラ | `dino.c:606-616` | 注視点固定のオービットカメラ(az/el/dist)、焦点距離 FL=1.8 |
| レイ生成 | `dino.c:625-631` | 横4ピクセルを1パケットに(`X4 = (0,1,2,3)`) |
| 個体境界球 | `dino.c:637-656` | 3個体の球とレイの交差。全ミスなら即スキップ |
| プライマリマーチ | `dino.c:658-679` | `buildList` で絞ったリストに対し **64 iter** の sphere tracing。ステップ 0.92 倍の緩和、ヒット閾値は距離比例 |
| 法線・AO | `dino.c:688-703` | **四面体法線**(4 サンプル)、AO は法線方向 1 tap |
| albedo | `dino.c:705-718` | 地面 `groundAlbedo()`(`:466-477`、チェッカー + grime ノイズ、SCROLL でスクロール)と恐竜 `dinoAlbedo()` を hit マスクで合成 |
| ライティング | `dino.c:720-741` | 拡散(太陽1灯)+ 半球アンビエント + Blinn-Phong スペキュラ(目 1.4 / 恐竜 0.35 / 床 0.05) |
| ソフトシャドウ | `dino.c:559-596` `softshadow()` | 3個体の境界球の min entry / max exit で**影レイ区間を統合**(相互影が付く)。12 iter、`min(res, 9·d/s)` の古典式 |
| 体表の環境反射 | `dino.c:743-769` | 反射方向の**解析スカイ + 地面チェッカー**をフレネル `F = 0.03 + 0.55·(1−cosθ)⁵` で加算(再マーチなし)。骨色部位(角・嘴・スパイク)は光沢 1.5 倍(`:764`) |
| 床反射(実マーチ) | `dino.c:770-847` | 地面ヒット点から Y 反転レイで群れを**再マーチ(30 iter)**、影なし簡易シェーディング。ミス時は雲入り解析スカイ。フレネル `kR = 0.14 + 0.60·(1−cosθ)⁵` でブレンド、影の中は減光(`:843`) |
| 霧・空 | `dino.c:849-856` | 距離²ベースの霧、ミスピクセルは `skyCol()`(`:438-464`、グラデ + 太陽グレア + 2オクターブ雲) |
| 出力 | `dino.c:858-864` | ガンマ 2.0(= sqrt)、RGBA8 パックして `FB` に SIMD ストア |

### 9. テクスチャ / 種別の見た目(`dino.c:479-557`)

- `dinoMasks()`(`:479-492`): ヒット点から最寄りの個体境界球で種マスク m0/m1/m2 を作る。
- `dinoAlbedo()`(`:494-557`): 種別パレット — 獣脚類 = 緑 + 背縞、ステゴ = 暖褐 + **背板の赤橙グラデ**(板ジオメトリを解析再評価する厳密判定 `PLT[]` `:513-526`。高さパラメータ h からグラデーション)、トリケラ = 灰緑 + フリル同心リング + 暖色リム。共通で 2 オクターブ値ノイズ変調 + 腹の明色化(`:548-552`)。

## 使われている技術まとめ

- **WebAssembly SIMD128** — `<wasm_simd128.h>` の intrinsics を直接使用(f32x4 演算、`bitselect`、`any_true`、`i32x4_trunc_sat` など)
- **フリースタンディング C** — `-nostdlib -Wl,--no-entry`、libc/libm なし。`__builtin_sqrtf` 等 + 自前 sin 近似
- **Sphere tracing (SDF レイマーチング)** — smooth min、四面体法線、SDF ベースのソフトシャドウ / AO
- **異方性カプセル SDF** — Lipschitz ≤ 1 を保った軸別スケール(本デモの独自ポイント)
- **4-wide レイパケット** — SoA + レーンマスクによる分岐レス SIMD 化
- **解析的アニメーション** — ステートレスなスキュー正弦の群れ運動、転がり接触拘束の歩行、2ボーン IK
- **ゼロコピー描画** — wasm メモリ上の `ImageData` を `putImageData` → cover-fit 拡大
- **適応解像度** — 実測フレーム時間の EMA による 5 段階自動調整

## 調整ポイント(改修の入口)

- **群れ**: `animate()` の `kin(t, x0, D, ω, φ, a)`(`dino.c:411-413`)。D·ω·(1+a) がバースト速度を決める
- **歩幅/ケイデンス**: 各種の `A`(`qleg` の amp / `tleg` 内の 0.315)と `ds` を変えたらケイデンス係数を π·ds/A に合わせて更新(コメント付きの箇所)
- **速度/品質**: プライマリ 64 / 反射 30 / 影 12 の各 iter 数。反射の有無は `if (any(gm))` ブロック(`dino.c:770`)
- **体型**: 各種ビルダー(`theropod` / `stego` / `trice`)の `cone` / `coneA` 列
- **テクスチャ**: `dinoAlbedo()` のパレットと領域ルール(ローカル座標ベース)
- **ブレンド硬さ**: `SK`(`dino.c:93`。変えたらバウンド計算との整合に注意)
- **解像度段階**: `template.html:70` の `LV` 配列と切替閾値(`:143-144`)

## GitHub Pages への公開

`.github/workflows/pages.yml` が main ブランチへの push をトリガに、CI 上で
`build.sh` を実行して `dino-herd.html` を `index.html` として GitHub Pages に
デプロイします。初回のみリポジトリの Settings → Pages で
Source を **GitHub Actions** にしてください(workflow 側でも自動有効化を試みます)。

公開 URL: <https://reki2000.github.io/wasm-raytrace/>

## ライセンス

MIT License(`LICENSE` 参照)
