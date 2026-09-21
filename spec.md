# Software Design & Development Specification — KDAA Acoustic Front-End (AFE)

Status: Draft 1.0 · Owner: twopercenz (CNN + local development) · Audience: implementer (human or AI), teammates
Document type: **spec-driven development (SDD)** — implementation follows this document; requirements are
traceable by ID (FR/NFR/AC). This document is the source of truth; code conforms to it, not the reverse.

---

## 1. Purpose & Scope

The AFE is the on-device acoustic classifier of the KDAA pipeline. It consumes a **log-mel
spectrogram per keystroke** and produces, for each keystroke, a **top-N distribution over
symbols** (33 Hangul base jamo + special-key tokens). It emits these to the LM Server, which
performs all syllable composition and language correction.

### 1.1 In scope
- CNN classification of one keystroke's log-mel into 38 symbol classes.
- Top-N alternatives with normalized probabilities.
- Special keys handled as ordinary token classes (no special-case logic).
- Training and evaluation of the classifier.
- Export of the trained model to the ESP32-S3 runtime (int8).

### 1.2 Out of scope (owned elsewhere)
- **Mel spectrogram extraction and segmentation/onset detection** — owned by the Mel component.
  The AFE never touches raw audio; it receives pre-segmented, pre-normalized log-mel tensors.
- **Syllable composition, Dubeolsik automaton, candidate generation, beam search, LM rerank** — owned
  by the LM Server.
- ESP32-S3 firmware / audio capture / recorder hardware.

### 1.3 Position in the system
```
Recorder(48kHz) ─► [Mel component: onset detect → 100ms window → log-mel] ─► [AFE: CNN → top-N symbols]
                                                                                     │
                                                                                     ▼
                                                          [LM Server: automaton → beam → text]
```

---

## 2. Design Decisions (resolved)

Recorded ADR-style; each is FROZEN unless this document is revised.

- **D1 — Runtime & quantization.** Deploy target is ESP32-S3 via **TensorFlow Lite Micro**, model
  quantized to **int8** (post-training quantization). Rationale: TFLite Micro is the most portable
  microcontroller inference path; int8 fits the memory/latency budget (NFR-1).
- **D2 — Sample rate.** **48 kHz**, mono. The microphone will support it (confirmed). All training and
  deployment log-mel use the identical rate.
- **D3 — Special keys.** Space / Backspace / Shift / CapsLock are **ordinary token classes**, not a
  separate head or special processing. The classifier output space is a single flat set of 38 symbols.
  Any downstream interpretation (e.g., Shift → tense consonant) is the LM Server's job.
- **D4 — Segmentation ownership.** The Mel component owns onset detection **and** the 100 ms windowing.
  The AFE receives one log-mel per keystroke plus its `onset_s`.
- **D5 — Output schema.** The AFE→LM contract is `kdaa.afe.v1` (§5), agreed with the LM Server team.
- **D6 — No composition in AFE.** The AFE emits symbols and probabilities only. `src/hangul.py`'s
  composition/beam-search exists solely for standalone AFE demos/evaluation, never in the live pipeline.

---

## 3. Interfaces (contracts)

The AFE is a pure function between two frozen contracts.

### 3.1 Input contract (Mel → AFE)

Provided by the Mel component. Extraction parameters are frozen and must be **identical for training
and deployment**; a mismatch causes domain shift and silent accuracy loss.

Frozen extraction parameters:
- `sample_rate = 48000`
- `n_fft = 1024`
- `win = 8 ms` (384 samples @48k), `hop = 2 ms` (96 samples @48k)
- `n_mels = 64`, `fmin = 200 Hz`, `fmax = 20000 Hz`
- clip length = `100 ms` (onset −5 ms … +95 ms; press + release)
- time frames fixed to `48` (center-crop if longer, edge-pad if shorter)
- value = `log(mel + 1e-6)` then **per-clip z-score** (mean 0, std 1)
- output = `float32`, **shape `(1, 64, 48)`**

Per-keystroke record (runtime; streamed or batched). Special-key keystrokes are included — the Mel
component does not distinguish jamo from special keys, it emits a record for every keystroke:
```json
{ "idx": 0, "onset_s": 1.234, "logmel": "<float32[64][48]>" }
```
Session (batch) envelope:
```json
{
  "mel_spec_version": "1.0",
  "param_hash": "ab12cd34",
  "sample_rate": 48000, "n_mels": 64, "frames": 48,
  "session_id": "p1_near_s0",
  "records": [ { "idx": 0, "onset_s": 1.234, "logmel": "..." }, "..." ]
}
```
Rules:
- `logmel` is the **final** tensor (normalization already applied). AFE adds no preprocessing.
- `idx` is time-ordered and contiguous; no gaps.
- `onset_s` is passed through unchanged to the output; the AFE does not compute onsets.
- The AFE validates `(n_mels, frames)` against the frozen regime and **fails loudly** on mismatch (AC-8).

For **training**, the Mel component supplies log-mel either as (A, preferred) pre-computed tensors keyed
by `clip_id` (e.g. `mel/<clip_id>.npy`) joined to the label table, or (B) a shared extractor library
pinned to the parameters above. Either way `param_hash` must match deployment.

### 3.2 Output contract (AFE → LM Server): `kdaa.afe.v1`

Machine-checkable schema lives at `schemas/kdaa_afe_v1.schema.json`. Canonical shape:
```json
{
  "schema": "kdaa.afe.v1",
  "sample_rate": 48000,
  "session_id": "p1_near_s0",
  "lang": "ko",
  "tokens": [
    {
      "idx": 0,
      "onset_s": 1.234,
      "keytype": "normal",
      "alts": [
        {"jamo": "ㄱ", "p": 0.72},
        {"jamo": "ㄷ", "p": 0.16},
        {"jamo": "ㅋ", "p": 0.12}
      ]
    },
    { "idx": 1, "onset_s": 1.402, "keytype": "space",
      "alts": [ {"jamo": "<sp>", "p": 1.0} ] }
  ],
  "meta": { "model_id": "smallcnn-int8-v1", "topN": 5, "input_shape": [1,64,48], "calibrated": true }
}
```
Field semantics:
- `tokens[]` — one element per keystroke, time-ordered, contiguous.
  - `idx` — 0-based sequence index.
  - `onset_s` — keystroke time in seconds, passed through from the input.
  - `keytype` — one of `normal | space | backspace | shift | caps | other`, **derived** from the top-1
    symbol (`<sp>`→space, `<bs>`→backspace, `<shift>`→shift, `<caps>`→caps, `<other>`→other, else normal).
  - `alts[]` — top-N candidates, probability-descending, **renormalized to sum to 1**.
    - `jamo` — the symbol string; a base jamo for normal keys, or a special token (`<sp>`, …).
    - `p` — probability (temperature-scaled if `meta.calibrated` is true).
- `meta` — `model_id`, `topN`, `input_shape` (the agreed log-mel regime), `calibrated` (bool).

Boundary rule: the AFE never outputs syllables, automaton state, or dictionary candidates.

---

## 4. Label Space (data contract)

Single flat classifier output = **38 classes**.
- 33 base jamo (from `src/hangul.py::LABELS`):
  - consonants (14): ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ
  - tense (5): ㄲㄸㅃㅆㅉ · vowels (12): ㅏㅐㅑㅓㅔㅕㅗㅛㅜㅠㅡㅣ · shift vowels (2): ㅒㅖ
- 5 special tokens: `<sp>` space, `<bs>` backspace, `<shift>` shift, `<caps>` caps, `<other>`.

Ground-truth labels come from the recording metadata (keylogger), not from the Mel component. The label
table adds a `keytype` column so special keystrokes are labeled with their token; `jamo` is meaningful
only when `keytype == normal`.

---

## 5. Functional Requirements

- **FR-1 Input consumption.** Load the Mel input contract (§3.1); validate the log-mel regime; fail loudly
  on mismatch. AFE performs no audio processing or mel extraction.
- **FR-2 Classification.** A single-head CNN maps `(1,64,48)` → 38-class logits. Architecture must fit the
  deployment budget (NFR-1); a compact CNN is the baseline.
- **FR-3 Top-N with probabilities.** Apply softmax (optionally temperature-scaled), take the top-N symbols
  per keystroke, renormalize their probabilities to sum to 1, and populate `alts[]`.
- **FR-4 Keytype derivation.** Set `keytype` from the top-1 symbol per the mapping in §3.2.
- **FR-5 Output emission.** Produce a schema-valid `kdaa.afe.v1` document from a session of log-mels and
  their onsets.
- **FR-6 Training.** Train the classifier from (log-mel, label) pairs. Split **by session** (never split a
  recording session across train/test); expose a by-participant split for cross-user generalization.
- **FR-7 Evaluation.** Report top-1 and top-N accuracy, **top-N coverage** (fraction where the true symbol
  is within top-N — this is the LM's performance ceiling), per-scenario accuracy (near/far/noise), a
  confusion matrix, keytype accuracy, and, when calibration is used, **ECE**. Metrics are emitted as
  grayscale figures plus a metrics file.
- **FR-8 Calibration (optional).** Fit a single temperature on a validation split so probabilities are
  calibrated; apply it at inference and set `meta.calibrated = true`. Gated on whether the LM Server relies
  on probability magnitudes (open item O-Q4).
- **FR-9 Export.** Convert the trained model PyTorch → ONNX → TFLite int8 (PTQ with a representative set).
  Emit the input quantization `(scale, zero_point)` and hand it to the Mel component (§3.1, on-device the
  Mel output is quantized to int8 with these before entering the CNN). Emit the label list and input regime
  as deployment metadata.
- **FR-10 Streaming (later).** Given per-keystroke log-mel arriving as a stream, emit tokens incrementally
  within the latency budget. Batch (session) mode ships first.

---

## 6. Non-Functional Requirements

- **NFR-1 Device budget.** ESP32-S3 class (≈512 KB SRAM, ≈8 MB PSRAM, no GPU). CNN int8 weights **≤ ~1 MB**;
  CNN inference **≤ 30 ms per keystroke**. (Mel extraction cost is the Mel component's budget, not AFE's.)
- **NFR-2 Regime consistency.** Training and deployment use the identical log-mel regime (§3.1), verified by
  `param_hash`.
- **NFR-3 Accuracy targets (provisional; refine after data).** Near condition top-1 ≥ 85%, top-5 ≥ 95%.
  Far/noise are allowed to degrade and are reported separately.
- **NFR-4 Reproducibility.** Fixed seeds, fixed session-wise split, saved training curves and metrics.
- **NFR-5 Determinism.** Identical input yields identical output.
- **NFR-6 Fail-safe.** A log-mel whose shape ≠ the agreed regime must raise, never silently mis-infer (AC-8).

---

## 7. Data Requirements

- Session label CSV columns: `onset_s, jamo, keytype, shift, scenario, participant`.
  - `keytype ∈ {normal, space, backspace, shift, caps, other}`.
  - `jamo` meaningful only for `normal`; special keys carry `<sp>/<bs>/<shift>/<caps>`.
- The keylogger recorder already captures physical keys, so Space/Backspace/Shift/Caps timestamps are
  available — extend it to record `keytype`.
- The prompt corpus must include enough Space / Backspace / Shift usage to balance the special-token classes.
- Coverage target: ≥ 25 keystrokes per jamo (per participant), plus a minimum per special token.

---

## 8. Module Plan (files)

- `src/labels.py` — 38-symbol space (33 jamo + 5 special), `keytype_of`, index maps. **[done, stub]**
- `src/melio.py` — Mel input adapter and regime validation (`MelSession.check`). **[done, stub]**
- `src/model.py` — single-head CNN `SmallCNN(num_classes=38)`; keep a heavier `CoAtNetLite` option for
  offline comparison only. **[extend]**
- `src/infer.py` — log-mel → `kdaa.afe.v1` (top-N, renormalization, keytype). **[done, core]**
- `src/calibrate.py` — temperature fit + ECE. **[stub]**
- `src/export_esp32.py` — ONNX → TFLite int8 PTQ; emit `(scale, zero_point)`. **[stub]**
- `src/train.py`, `src/evaluate.py` — consume log-mel; add keytype accuracy, top-N coverage, ECE. **[extend]**
- `schemas/kdaa_afe_v1.schema.json` — output schema. **[done]**
- `tests/test_infer_schema.py` — output conforms to schema; alts sum to 1; special-token keytype. **[done]**
- `src/features.py`, `src/segment.py` — **external** (Mel-owned); retained for reference only.

---

## 9. Configuration

```yaml
labels:
  num_classes: 38                 # 33 jamo + 5 special tokens
input:                            # frozen log-mel regime (consumed, not produced)
  n_mels: 64
  frames: 48
  sample_rate: 48000
  normalization: per_clip_zscore
infer:
  top_n: 5
  calibrate: true
  temperature: 1.0                # set by calibrate.py
export:
  runtime: tflite_int8
  calib_clips: 300                # representative samples for PTQ
train:
  split: session                  # session | participant | random
```

---

## 10. Acceptance Criteria (traceable)

- **AC-1** (FR-5) Session log-mel input yields a document that validates against
  `schemas/kdaa_afe_v1.schema.json`.
- **AC-2** (FR-5) Token count equals keystroke count (special keys included); order preserved; `onset_s`
  passed through unchanged.
- **AC-3** (NFR-3, FR-7) Near-condition top-1 meets target; **top-5 coverage ≥ 95%**.
- **AC-4** (FR-4/FR-7) Keytype accuracy ≥ 95% (Space/Backspace prioritized).
- **AC-5** (FR-8) When calibration is on, ECE ≤ 0.05.
- **AC-6** (FR-9) int8 export loses ≤ 2 percentage points top-1 vs. the float reference; size and latency
  within NFR-1.
- **AC-7** (NFR-5) Deterministic output for identical input.
- **AC-8** (NFR-6) Log-mel with a non-conforming shape raises rather than mis-infers.
- **AC-9** (FR-3) `alts[].p` sums to 1 (± 1e-6) for every token.

---

## 11. Work Breakdown (tickets)

Each ticket: file · goal · depends · done-when. Ordered by dependency.

- **T1 Label space** — `src/labels.py` — 38 classes + `keytype_of`. Done: `num_classes()==38`, schema test
  green. **[done]**
- **T2 Mel adapter** — `src/melio.py` — load Mel contract, validate regime. Depends: O-O3 (transfer format).
  Done: a real Mel sample loads; shape/hash validated.
- **T3 CNN head** — `src/model.py` — `SmallCNN(38)`, input `(B,1,64,48)`. Done: forward shape `(B,38)`;
  parameter/complexity within NFR-1.
- **T4 Training on log-mel** — `src/train.py`, `src/dataset.py` — dataset yields (log-mel, label); session
  split retained. Depends: T2, T8, O-O4. Done: 1 epoch on sample log-mel; checkpoint saved.
- **T5 Inference → schema** — `src/infer.py` — top-N, renormalize, keytype. Done: `test_infer_schema.py`
  green; wire the real model after T4. **[done, core]**
- **T6 Evaluation** — `src/evaluate.py`, `src/calibrate.py` — top-1/N, keytype acc, top-N coverage, ECE,
  grayscale figures. Done: metrics + figures; AC-3/AC-4/AC-5 measured.
- **T7 Calibration** — `src/calibrate.py` — fit temperature, pass to `infer`. Depends: O-Q4. Done: ECE ≤ 0.05.
- **T8 keytype labels** — recorder/corpus/metadata — record `keytype`; balance special tokens. Done:
  metadata has `keytype`; minimum samples per special token.
- **T9 ESP32 export** — `src/export_esp32.py` — ONNX → TFLite int8; emit `(scale, zero_point)` to Mel.
  Depends: T3/T4, TF+onnx2tf toolchain. Done: `model_int8.tflite`; AC-6 met.
- **T10 Streaming** — `src/infer.py` — incremental emission. Depends: batch path stable. Done: within latency.

Dependency sketch:
```
T1 ─┐
T2 ─┼─► T4 ─► T6 ─► T7
T3 ─┘        └─► T9
T8 ─► T4      T5 (real model after T4)      T10 last
```
Recommended next: **T8** (labels — needed before training) → **T2** (on Mel sample format) → **T4** → T6 → T9.

---

## 12. Risks

- **Regime drift.** Mel parameters change mid-project → model input shape and retraining. Mitigation: freeze
  §3.1 early, enforce `param_hash`.
- **Quantization distortion.** int8 warps top-N probabilities → mitigate with calibration; re-verify after
  export (AC-6).
- **Special-key imbalance.** Too few Space/BS/Shift samples → weak special-token classes. Mitigation: corpus
  balancing (T8).
- **Train/deploy mel mismatch.** Different extraction offline vs on-device → pin the same regime/library with
  the Mel component.

---

## 13. Open Items

Non-blocking; settle during implementation.
- **O-Q4 Calibration need.** Does the LM Server use probability magnitudes? If yes, ship FR-8.
- **O-O3 Transfer format.** Mel → AFE as real-time stream vs. session file; serialization (npy/binary/JSON).
- **O-O4 Training log-mel provisioning.** Pre-computed tensors (preferred) vs. shared extractor.
- **O-O5 Quantization hand-off.** When AFE shares `(scale, zero_point)` with Mel after export.
- **O-Q6 Data schedule / special-key corpus.**
- **O-Q7 Final accuracy targets.**
- **O-Q8 v2 Latin/English merge timing** (after Korean v1 is stable): extend the label space, add a `lang`
  branch, collect data.

---

## 14. Glossary

- **AFE** — Acoustic Front-End: the CNN component specified here.
- **log-mel** — log-magnitude mel-scaled spectrogram; the AFE's input, one per keystroke.
- **symbol** — a classifier output: a base jamo or a special token.
- **keytype** — coarse category of a keystroke derived from its symbol.
- **top-N coverage** — fraction of keystrokes whose true symbol appears in the top-N; upper bound on what the
  LM Server can recover.
- **PTQ** — post-training quantization (float model → int8 using a representative dataset).
- **session-wise split** — train/test separation by recording session, to avoid optimistic bias.