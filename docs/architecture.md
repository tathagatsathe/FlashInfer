# FlashInfer Architecture

FlashInfer is a minimal GPT-2 inference engine built from scratch in PyTorch. It loads HuggingFace GPT-2 weights and runs text generation through a custom model implementation with KV-cache incremental decode.

---

## Project Structure

```mermaid
flowchart TB
    subgraph entry [Entry Points]
        CLI["examples/generate.py"]
        API["InferenceEngine.generate()"]
    end

    subgraph engine [Engine Layer]
        IE["flashinfer/engine/generator.py"]
        Sampler["flashinfer/sampling/sampler.py"]
    end

    subgraph model [Model Layer]
        GPT2["flashinfer/model/gpt2.py"]
        Cache["flashinfer/cache/kv_cache.py"]
    end

    subgraph layers [Layer Primitives]
        Emb["embeddings.py"]
        Attn["attention.py"]
        MLP["mlp.py"]
        Norm["norm.py"]
    end

    subgraph external [External Dependencies]
        HFWeights["HuggingFace GPT-2 weights"]
        HFTokenizer["HuggingFace tokenizer"]
        Transformers["transformers library"]
    end

    subgraph support [Support]
        Loader["flashinfer/weights/loader.py"]
        Tests["tests/"]
    end

    CLI --> IE
    API --> IE
    IE --> Sampler
    IE --> GPT2
    IE --> Cache
    IE --> HFTokenizer
    IE --> Loader
    Loader --> HFWeights
    Loader --> Transformers
    GPT2 --> Emb
    GPT2 --> Attn
    GPT2 --> MLP
    GPT2 --> Norm
    GPT2 --> Cache
    Attn --> Cache
    Tests --> GPT2
    Tests --> IE
```

| Module | Role |
|--------|------|
| [`flashinfer/engine/generator.py`](../flashinfer/engine/generator.py) | Orchestrates tokenization, prefill, decode loop, decoding |
| [`flashinfer/model/gpt2.py`](../flashinfer/model/gpt2.py) | Full GPT-2 stack: embeddings → 12 blocks → LM head |
| [`flashinfer/cache/kv_cache.py`](../flashinfer/cache/kv_cache.py) | Pre-allocated K/V buffers for incremental decode |
| [`flashinfer/layers/`](../flashinfer/layers/) | Reusable building blocks (attention, MLP, norm, embeddings) |
| [`flashinfer/weights/loader.py`](../flashinfer/weights/loader.py) | Maps HuggingFace Conv1D weights → PyTorch Linear |
| [`flashinfer/sampling/sampler.py`](../flashinfer/sampling/sampler.py) | Greedy, temperature, top-k, top-p sampling |

---

## Initialization Flow

Model loading happens once via `InferenceEngine.from_pretrained()`.

```mermaid
sequenceDiagram
    participant User
    participant Engine as InferenceEngine
    participant Loader as weights/loader.py
    participant HF as HuggingFace Hub
    participant Model as GPT2Model

    User->>Engine: from_pretrained("gpt2")
    Engine->>Loader: config_from_hf("gpt2")
    Loader->>HF: download config
    HF-->>Loader: GPT2Config
    Loader-->>Engine: vocab, layers, heads, dims

    Engine->>Model: GPT2Model.from_config(config)
    Engine->>Loader: load_hf_weights(model, "gpt2")
    Loader->>HF: download state_dict
    HF-->>Loader: transformer.wte, h.0.attn.c_attn, ...
    Loader->>Model: map Conv1D weights to Linear layers

    Engine->>HF: AutoTokenizer.from_pretrained("gpt2")
    HF-->>Engine: tokenizer
    Engine-->>User: ready InferenceEngine
```

Weight mapping example:

```
HF:   transformer.h.0.attn.c_attn.weight  (768 × 2304)
         ↓ transpose
Ours: blocks[0].attn.qkv.weight           (2304 × 768)
```

---

## Generation Execution Flow

This is the main runtime path used by `generate()` and the CLI.

```mermaid
flowchart TD
    Start([User prompt string]) --> Tokenize["Tokenizer.encode()"]
    Tokenize --> TokenIds["input_ids shape: 1 x L"]

    TokenIds --> CreateCache["KVCache.empty()"]
    CreateCache --> Prefill["model.forward_prefill(input_ids, cache)"]

    Prefill --> PrefillLogits["logits shape: 1 x L x 50257"]
    PrefillLogits --> LastLogits["next_logits = logits[0, -1, :]"]

    LastLogits --> SampleLoop{Decode loop}
    SampleLoop --> Sample["sample_next_token()"]
    Sample --> AppendToken["Append token to sequence"]
    AppendToken --> CheckEOS{EOS or max tokens?}
    CheckEOS -->|Yes| DecodeOut["tokenizer.decode()"]
    CheckEOS -->|No| DecodeStep["model.forward_decode([[token]], cache)"]
    DecodeStep --> DecodeLogits["logits shape: 1 x 1 x 50257"]
    DecodeLogits --> LastLogits
    DecodeOut --> Output([Generated text])
```

### Phase Breakdown

| Phase | Input | Model call | Output | Cache state |
|-------|-------|------------|--------|-------------|
| **Prefill** | Full prompt `(1, L)` | `forward_prefill()` | Logits for all L positions | K/V stored for positions `0..L-1`, `seq_len = L` |
| **Decode** (×N) | Single token `(1, 1)` | `forward_decode()` | Logits for 1 position | K/V appended, `seq_len += 1` |

Prefill runs once over the prompt. Each decode step processes a single new token using cached K/V tensors — **O(n) per step** vs **O(n²)** when re-forwarding the full sequence.

---

## GPT-2 Model Internal Dataflow

```mermaid
flowchart LR
    subgraph input [Input]
        IDs["input_ids"]
    end

    subgraph embed [Embeddings]
        WTE["token_emb wte"]
        WPE["pos_emb wpe"]
        Sum["token + position"]
    end

    subgraph block [GPT2Block x12]
        LN1["LayerNorm ln_1"]
        Attn["CausalSelfAttention"]
        Res1["+ residual"]
        LN2["LayerNorm ln_2"]
        MLPBlock["GPT2MLP"]
        Res2["+ residual"]
    end

    subgraph head [Output Head]
        LNF["LayerNorm ln_f"]
        LM["lm_head Linear"]
    end

    IDs --> WTE
    IDs --> WPE
    WTE --> Sum
    WPE --> Sum
    Sum --> LN1 --> Attn --> Res1
    Sum --> Res1
    Res1 --> LN2 --> MLPBlock --> Res2
    Res1 --> Res2
    Res2 --> LNF --> LM
    LM --> Logits["logits vocab_size=50257"]
```

### Single Transformer Block

```mermaid
flowchart TB
    X["hidden states x"] --> LN1["LayerNorm"]
    LN1 --> QKV["Linear qkv → split Q, K, V"]
    QKV --> AttnMatmul["softmax(QK^T / sqrt(d)) V"]
    AttnMatmul --> Proj["Linear out_proj"]
    Proj --> Add1["x + attn_out"]

    Add1 --> LN2["LayerNorm"]
    LN2 --> FC["Linear fc 768→3072"]
    FC --> GELU["gelu_new activation"]
    GELU --> ProjMLP["Linear proj 3072→768"]
    ProjMLP --> Add2["+ residual"]
    Add2 --> Out["block output"]
```

Tensor shapes for `gpt2` (124M):

| Tensor | Shape |
|--------|-------|
| `input_ids` | `(batch, seq_len)` |
| Hidden states | `(batch, seq_len, 768)` |
| Q, K, V per head | `(batch, 12, seq_len, 64)` |
| MLP intermediate | `(batch, seq_len, 3072)` |
| Logits | `(batch, seq_len, 50257)` |

---

## KV Cache in Attention

```mermaid
flowchart TD
    subgraph prefillPath [Prefill seq_len greater than 1]
        P1["Compute Q, K, V for all L tokens"]
        P2["Apply causal mask"]
        P3["Attention + store K,V in cache"]
        P1 --> P2 --> P3
    end

    subgraph decodePath [Decode seq_len equals 1]
        D1["Compute Q, K, V for 1 new token"]
        D2["Append K,V to cache at position seq_len"]
        D3["Attend: Q_new @ K_cache_full"]
        D1 --> D2 --> D3
    end

    subgraph cacheStore [KVCache per layer]
        Keys["keys: batch x heads x max_len x head_dim"]
        Values["values: batch x heads x max_len x head_dim"]
        SeqLen["seq_len counter"]
    end

    P3 --> cacheStore
    D2 --> cacheStore
    cacheStore --> D3
```

Cache layout (12 layers for gpt2):

```
KVCache
├── layers[0].keys   (1, 12, 1024, 64)
├── layers[0].values (1, 12, 1024, 64)
├── layers[1].keys   ...
├── ...
└── seq_len = current number of tokens processed
```

---

## Sampling Pipeline

```mermaid
flowchart LR
    Logits["last-position logits 50257"] --> TempCheck{temperature <= 0?}
    TempCheck -->|Yes| Greedy["argmax"]
    TempCheck -->|No| Scale["logits / temperature"]
    Scale --> TopK{top_k set?}
    TopK -->|Yes| MaskK["mask below k-th largest"]
    TopK -->|No| TopP
    MaskK --> TopP{top_p set?}
    TopP -->|Yes| Nucleus["nucleus truncation"]
    TopP -->|No| Softmax
    Nucleus --> Softmax["softmax → probabilities"]
    Softmax --> Multi["multinomial sample"]
    Greedy --> TokenId["next token id"]
    Multi --> TokenId
```

---

## Dual Forward Paths

The project maintains two model paths:

```mermaid
flowchart TB
    subgraph production [Production Path - used by generate]
        Prefill["forward_prefill()"]
        Decode["forward_decode()"]
        KV["KVCache"]
        Prefill --> KV
        Decode --> KV
    end

    subgraph testing [Test Path - parity validation]
        FullForward["forward() full sequence"]
        NoCache["No KV cache"]
        FullForward --> NoCache
    end

    HF["HuggingFace reference logits"] --> FullForward
    FullForward --> ParityTests["tests/test_gpt2_parity.py"]
    Prefill --> KVTests["tests/test_kv_cache.py"]
    Decode --> KVTests
    KVTests --> FullForward
```

- **`forward()`** — full-sequence pass, no cache. Used by parity tests against HuggingFace.
- **`forward_prefill()` / `forward_decode()`** — production path with KV cache. Verified to match `forward()` step-by-step.

---

## End-to-End Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER / CLI                               │
│   "The capital of France is"  →  examples/generate.py           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    InferenceEngine                              │
│  1. Tokenize prompt → [464, 3139, 286, 4881, 318]              │
│  2. Create KVCache (12 layers × pre-allocated buffers)          │
│  3. PREFILL: forward_prefill → logits[-1] → sample token        │
│  4. DECODE loop: forward_decode → logits[-1] → sample token     │
│  5. Decode token ids → text                                     │
└──────┬──────────────────┬──────────────────┬────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│ GPT2Model   │   │ KVCache      │   │ sample_next_token│
│             │   │              │   │ (greedy/temp/    │
│ Embeddings  │◄─►│ K/V per layer│   │  top-k/top-p)   │
│ 12× Block   │   │ seq_len      │   └─────────────────┘
│ ln_f        │   └──────────────┘
│ lm_head     │
└─────────────┘
       ▲
       │ weights loaded once at startup
┌──────┴──────┐
│ HF gpt2     │
│ checkpoint  │
└─────────────┘
```

---

## Test Architecture

```mermaid
flowchart LR
    subgraph tests [Test Suite]
        T1["test_gpt2_parity.py"]
        T2["test_kv_cache.py"]
        T3["test_sampling.py"]
        T4["test_layers.py"]
    end

    T1 -->|"forward() vs HF"| Model
    T2 -->|"prefill/decode vs forward()"| Model
    T3 --> Sampler
    T4 --> Layers

    HFRef["transformers GPT2LMHeadModel"] --> T1
```

| Test file | What it validates |
|-----------|-------------------|
| `test_gpt2_parity.py` | Custom model logits match HuggingFace reference |
| `test_kv_cache.py` | Cached prefill/decode matches full forward path |
| `test_sampling.py` | top-k, top-p, and greedy sampling behavior |
| `test_layers.py` | Output shapes and causal mask correctness |

---

## Summary

FlashInfer follows a classic **prefill + decode** LLM serving pattern:

1. **Load once** — HuggingFace weights mapped into a custom PyTorch GPT-2
2. **Prefill once** — Process the full prompt, populate KV cache
3. **Decode many times** — One token per step using cached K/V
4. **Sample** — Convert logits → next token (greedy or stochastic)
5. **Repeat** until EOS or `max_new_tokens`

The design separates concerns cleanly: **layers** (math), **model** (architecture), **cache** (performance), **engine** (orchestration), **sampling** (decoding strategy).
