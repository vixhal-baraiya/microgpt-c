# Building GPT from Scratch in Pure C (Step-by-Step)

## Introduction

In this comprehensive guide, we'll build a working GPT (Generative Pre-trained Transformer) model entirely from scratch using pure C, no external libraries, no dependencies, just raw computational power.

Let's build something remarkable together.

---

## Part 1: Random Number Generation

Before we can train any neural network, we need randomness for weight initialization and sampling. Python has `random`, but we're in C, so we'll implement our own.

### Understanding Pseudo-Random Number Generation

We'll use the **xorshift algorithm** a fast, simple PRNG perfect for our needs. The algorithm maintains an internal state and transforms it through XOR and bit-shift operations to produce seemingly random numbers.

```c
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Our PRNG state - seed it like Python's random.seed(42) */
static unsigned long long rng_state = 42;

static unsigned long long rng_next(void) {
  rng_state ^= rng_state << 13;  // XOR with left-shifted self
  rng_state ^= rng_state >> 7;   // XOR with right-shifted self
  rng_state ^= rng_state << 17;  // XOR again
  return rng_state;
}
```

**What's happening?** The three XOR operations scramble the bits in a way that produces a sequence with good statistical properties. The shift amounts (13, 7, 17) are carefully chosen to maximize the period before repetition.

### Uniform Random Numbers

Now let's convert our random integers to floats between 0 and 1:

```c
static double rng_uniform(void) {
  // Take top 53 bits and divide by 2^53 to get [0,1)
  return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}
```

**Why 53 bits?** That's the precision of a double-precision float. We shift right by 11 to use the top 53 bits (64 - 11 = 53).

### Gaussian Random Numbers

Neural networks need normally distributed random values for weight initialization. We'll use the **Box-Muller transform** to convert uniform random numbers into Gaussian ones:

```c
static float rng_gauss(float mean, float std) {
  double u1 = rng_uniform();
  double u2 = rng_uniform();
  
  // Avoid log(0)
  if (u1 < 1e-30) u1 = 1e-30;
  
  // Box-Muller transform
  return mean + std * (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}
```

**The math:** Box-Muller takes two uniform random variables U₁ and U₂ and transforms them:
- Z = √(-2 ln U₁) × cos(2π U₂)

This Z follows a standard normal distribution! Scale by `std` and shift by `mean` to get any Gaussian you want.

### Shuffling Arrays

For training, we'll need to shuffle our dataset:

```c
static void shuffle_ints(int *arr, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(rng_uniform() * (i + 1));  // Random index from 0 to i
    // Swap arr[i] and arr[j]
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}
```

This is the **Fisher-Yates shuffle** it guarantees every permutation is equally likely.

---

## Part 2: Data Loading and Tokenization

GPT needs text data to learn from. We'll load documents and build a character-level tokenizer.

### Loading the Dataset

```c
#define MAX_DOCS 85000     // Maximum number of documents
#define MAX_DOC_LEN 512      // Maximum characters per document
#define MAX_CHARS 50      // Maximum unique characters

static char docs[MAX_DOCS][MAX_DOC_LEN];  // Our text corpus
static int num_docs = 0;

static void load_dataset(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", filename);
    exit(1);
  }
  
  char line[256];
  while (fgets(line, sizeof(line), f) && num_docs < MAX_DOCS) {
    // Remove trailing newlines
    int len = (int)strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = 0;
    
    if (len > 0) {
      strncpy(docs[num_docs], line, MAX_DOC_LEN - 1);
      docs[num_docs][MAX_DOC_LEN - 1] = 0;  // Ensure null termination
      num_docs++;
    }
  }
  fclose(f);
}
```

**Why this structure?** We store each line as a separate document. This lets us treat each line as an independent training sequence, which is perfect for learning patterns in short texts (like names, phrases, or code snippets).

### Building a Character-Level Tokenizer

Instead of word-based tokens, we'll use characters. This is simpler and works well for small models:

```c
static char uchars_arr[MAX_CHARS];  // Sorted unique characters
static int vocab_size, BOS, num_uchars = 0;

static int char_to_id(char c) {
  for (int i = 0; i < num_uchars; i++)
    if (uchars_arr[i] == c)
      return i;
  return -1;
}

static int cmp_char(const void *a, const void *b) {
  return *(const char *)a - *(const char *)b;
}

static void build_tokenizer(void) {
  int seen[256] = {0};  // Track which ASCII characters appear
  
  // Scan all documents for unique characters
  for (int d = 0; d < num_docs; d++)
    for (int i = 0; docs[d][i]; i++)
      seen[(unsigned char)docs[d][i]] = 1;
  
  // Collect unique characters
  for (int i = 0; i < 256; i++)
    if (seen[i])
      uchars_arr[num_uchars++] = (char)i;
  
  // Sort them for consistency
  qsort(uchars_arr, num_uchars, sizeof(char), cmp_char);
  
  // Add a special BOS (Beginning of Sequence) token
  BOS = num_uchars;
  vocab_size = num_uchars + 1;
}
```

**Why BOS?** The Beginning-of-Sequence token tells the model "start generating here." It's like a prompt that kicks off generation.

**Why sort?** It's not strictly necessary, but it makes debugging easier when characters have consistent IDs across runs.

---

## Part 3: Model Architecture - Hyperparameters

Now we define our GPT's structure. These are small values for a minimal model that trains quickly:

```c
#define N_EMBD 64        // Embedding dimension
#define N_HEAD 4         // Number of attention heads
#define N_LAYER 2        // Number of transformer layers
#define BLOCK_SIZE 32     // Context window (sequence length)
#define HEAD_DIM (N_EMBD / N_HEAD)  // 16/4 = 4 dimensions per head
#define MLP_DIM (4 * N_EMBD)        // 64 hidden units in feedforward
```

**Understanding the dimensions:**

- **N_EMBD (64)**: Each token is represented as a 64-dimensional vector. This is tiny compared to GPT-3's 12,288, but it's enough for learning simple patterns.

- **N_HEAD (4)**: Multi-head attention splits the embedding into 4 heads of 4 dimensions each. Each head can attend to different patterns.

- **N_LAYER (2)**: We'll use just two transformer block. Real GPT models stack dozens of these.

- **BLOCK_SIZE (32)**: We look at 8 tokens of context at once. GPT-3 uses 2048.

- **MLP_DIM (256)**: The feedforward network has 4× the embedding dimension, following standard transformer architecture.

---

## Part 4: Allocating Parameters and Gradients

Neural networks are just collections of learnable numbers (parameters). We need storage for:
1. The parameters themselves
2. Their gradients (for backpropagation)
3. Optimizer state (for Adam)

### Parameter Structure

```c
/* Token and position embeddings */
static float *wte, *d_wte;           // Token embedding weights & gradients
static float *wpe, *d_wpe;           // Position embedding weights & gradients
static float *lm_head, *d_lm_head;   // Language model head (output projection)

/* Per-layer parameters - we'll have N_LAYER sets of these */
static float *attn_wq[N_LAYER], *d_attn_wq[N_LAYER];  // Query projection
static float *attn_wk[N_LAYER], *d_attn_wk[N_LAYER];  // Key projection
static float *attn_wv[N_LAYER], *d_attn_wv[N_LAYER];  // Value projection
static float *attn_wo[N_LAYER], *d_attn_wo[N_LAYER];  // Output projection
static float *mlp_fc1[N_LAYER], *d_mlp_fc1[N_LAYER];  // MLP first layer
static float *mlp_fc2[N_LAYER], *d_mlp_fc2[N_LAYER];  // MLP second layer

/* Adam optimizer state - momentum and velocity for each parameter */
static float *adam_m_wte, *adam_v_wte;
static float *adam_m_wpe, *adam_v_wpe;
static float *adam_m_lm, *adam_v_lm;
static float *adam_m_wq[N_LAYER], *adam_v_wq[N_LAYER];
static float *adam_m_wk[N_LAYER], *adam_v_wk[N_LAYER];
static float *adam_m_wv[N_LAYER], *adam_v_wv[N_LAYER];
static float *adam_m_wo[N_LAYER], *adam_v_wo[N_LAYER];
static float *adam_m_fc1[N_LAYER], *adam_v_fc1[N_LAYER];
static float *adam_m_fc2[N_LAYER], *adam_v_fc2[N_LAYER];
```

**Why so many arrays?** Each transformation in the model needs its own weight matrix. The `d_` prefix means "derivative" (gradient), and `adam_m`/`adam_v` are for momentum-based optimization.

### Initialization Functions

```c
static int num_params = 0;

static float *make_param(int size, float std) {
  float *p = (float *)calloc(size, sizeof(float));
  
  // Initialize with Gaussian noise
  for (int i = 0; i < size; i++)
    p[i] = rng_gauss(0, std);
  
  num_params += size;
  return p;
}

static float *make_zero(int size) {
  return (float *)calloc(size, sizeof(float));  // Zero-initialized
}
```

**Why Gaussian initialization?** Random initialization breaks symmetry. If all weights started at zero, all neurons would learn the same thing. The standard deviation (0.02 typically) is chosen to keep activations in a good range.

### Initializing All Parameters

```c
static void init_params(void) {
  int es = vocab_size * N_EMBD;    // Embedding size
  int ps = BLOCK_SIZE * N_EMBD;    // Position embedding size
  int as = N_EMBD * N_EMBD;        // Attention matrix size
  int ms = MLP_DIM * N_EMBD;       // MLP matrix size
  
  // Token embeddings: maps each token ID to a vector
  wte = make_param(es, 0.02f);
  d_wte = make_zero(es);
  adam_m_wte = make_zero(es);
  adam_v_wte = make_zero(es);
  
  // Position embeddings: adds position information to each token
  wpe = make_param(ps, 0.02f);
  d_wpe = make_zero(ps);
  adam_m_wpe = make_zero(ps);
  adam_v_wpe = make_zero(ps);
  
  // Language model head: projects back to vocabulary for predictions
  lm_head = make_param(es, 0.02f);
  d_lm_head = make_zero(es);
  adam_m_lm = make_zero(es);
  adam_v_lm = make_zero(es);
  
  // Initialize each transformer layer
  for (int i = 0; i < N_LAYER; i++) {
    attn_wq[i] = make_param(as, 0.02f);
    d_attn_wq[i] = make_zero(as);
    adam_m_wq[i] = make_zero(as);
    adam_v_wq[i] = make_zero(as);
    
    attn_wk[i] = make_param(as, 0.02f);
    d_attn_wk[i] = make_zero(as);
    adam_m_wk[i] = make_zero(as);
    adam_v_wk[i] = make_zero(as);
    
    attn_wv[i] = make_param(as, 0.02f);
    d_attn_wv[i] = make_zero(as);
    adam_m_wv[i] = make_zero(as);
    adam_v_wv[i] = make_zero(as);
    
    // Output projection starts at 0 (following GPT-2 initialization)
    attn_wo[i] = make_param(as, 0.0f);
    d_attn_wo[i] = make_zero(as);
    adam_m_wo[i] = make_zero(as);
    adam_v_wo[i] = make_zero(as);
    
    mlp_fc1[i] = make_param(ms, 0.02f);
    d_mlp_fc1[i] = make_zero(ms);
    adam_m_fc1[i] = make_zero(ms);
    adam_v_fc1[i] = make_zero(ms);
    
    mlp_fc2[i] = make_param(ms, 0.0f);
    d_mlp_fc2[i] = make_zero(ms);
    adam_m_fc2[i] = make_zero(ms);
    adam_v_fc2[i] = make_zero(ms);
  }
  
  printf("num params: %d\n", num_params);
}
```

**Why initialize output projections to 0?** This is a GPT-2 trick called "zero initialization" for residual connections. It makes training more stable early on.

---

## Part 5: Activation Storage for Backpropagation

During the forward pass, we need to save intermediate values. During the backward pass, we'll use these to compute gradients.

```c
typedef struct {
  float x_embed[N_EMBD];                      // Initial embedding
  float rms_scale_init;                       // RMS norm scale
  float x_in[N_LAYER][N_EMBD];               // Layer input
  float xn_attn[N_LAYER][N_EMBD];            // Normalized input to attention
  float rms_scale_attn[N_LAYER];             // RMS scale for attention norm
  float q[N_LAYER][N_EMBD];                  // Query vectors
  float aw[N_LAYER][N_HEAD][BLOCK_SIZE];     // Attention weights
  float attn_out[N_LAYER][N_EMBD];           // Attention output
  float x_mid[N_LAYER][N_EMBD];              // After attention residual
  float xn_mlp[N_LAYER][N_EMBD];             // Normalized input to MLP
  float rms_scale_mlp[N_LAYER];              // RMS scale for MLP norm
  float mlp_pre[N_LAYER][MLP_DIM];           // MLP before activation
  float mlp_post[N_LAYER][MLP_DIM];          // MLP after activation
  float x_out[N_EMBD];                       // Final output
} PosActs;

static PosActs saved[BLOCK_SIZE];              // Save activations for each position
static float saved_probs[BLOCK_SIZE][MAX_CHARS + 1];  // Saved probability distributions
```

**Why save all this?** Backpropagation is the reverse of the forward pass. To compute how much each parameter contributed to the loss, we need to know what values flowed through during forward pass.

### KV Cache and Gradient Accumulators

```c
/* KV cache stores keys and values for all positions in the sequence */
static float kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static float kv_vals[N_LAYER][BLOCK_SIZE][N_EMBD];

/* Gradient accumulators for keys and values across positions */
static float dk_accum[N_LAYER][BLOCK_SIZE][N_EMBD];
static float dv_accum[N_LAYER][BLOCK_SIZE][N_EMBD];
```

**What's a KV cache?** In attention, each position queries all previous positions. We store the Key and Value projections for all positions so we don't recompute them. This is critical for efficient autoregressive generation.

---

## Part 6: Building Blocks - Forward Pass Operations

Now we implement the fundamental operations: linear layers, normalization, and softmax.

### Linear Transformation (Matrix Multiplication)

```c
static inline void linear_fwd(const float *restrict x, const float *restrict w,
                              int nout, int nin, float *restrict out) {
  // Compute out = W @ x, where W is (nout × nin) and x is (nin × 1)
  for (int r = 0; r < nout; r++) {
    float s = 0;
    const float *wr = w + r * nin;  // Pointer to row r of W
    
    // Dot product of row r with input x
    for (int c = 0; c < nin; c++)
      s += wr[c] * x[c];
    
    out[r] = s;
  }
}
```

**The math:** This is `y = Wx` where W is a matrix and x is a vector. Each output element is a dot product of one row of W with x.

**Why `restrict`?** This keyword tells the compiler that pointers don't alias (overlap), enabling better optimization.

**Why `inline`?** These functions are called many times; inlining avoids function call overhead.

### RMS Normalization

Instead of LayerNorm, we use RMSNorm (Root Mean Square Normalization), which is simpler and works just as well:

```c
static inline float rmsnorm_fwd(const float *x, int n, float *out) {
  // Compute mean square
  float ms = 0;
  for (int i = 0; i < n; i++)
    ms += x[i] * x[i];
  ms /= n;
  
  // Compute scaling factor
  float scale = 1.0f / sqrtf(ms + 1e-5f);
  
  // Normalize
  for (int i = 0; i < n; i++)
    out[i] = x[i] * scale;
  
  return scale;  // Return for use in backward pass
}
```

**The math:** RMSNorm computes:
```
RMS = √(mean(x²))
y = x / RMS
```

**Why normalize?** It keeps activations in a stable range, preventing exploding or vanishing gradients. The `1e-5` prevents division by zero.

**Why return scale?** We need it during backpropagation to compute gradients correctly.

### Softmax

Softmax converts logits to probabilities:

```c
static inline void softmax_fwd(const float *logits, int n, float *probs) {
  // Find maximum for numerical stability
  float mx = logits[0];
  for (int i = 1; i < n; i++)
    if (logits[i] > mx)
      mx = logits[i];
  
  // Compute exp(x - max) and sum
  float sum = 0;
  for (int i = 0; i < n; i++) {
    probs[i] = expf(logits[i] - mx);
    sum += probs[i];
  }
  
  // Normalize to get probabilities
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++)
    probs[i] *= inv;
}
```

**The math:**
```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

**Why subtract max?** Without it, `exp(x)` can overflow for large x. Subtracting the max shifts all values into a safe range. Mathematically, this doesn't change the result:
```
exp(x - max) / Σ exp(x - max) = exp(x) / Σ exp(x)
```

---

## Part 7: Backward Pass Building Blocks

Backpropagation requires derivatives of our forward operations. Let's implement them.

### Linear Layer Backward

The linear forward was `y = Wx`. The backward computes gradients w.r.t. both x and W:

```c
/* Gradient w.r.t. input: dx = W^T @ dout */
static inline void linear_bwd_x(const float *restrict w,
                                const float *restrict dout, int nout, int nin,
                                float *restrict dx) {
  for (int c = 0; c < nin; c++) {
    float s = 0;
    for (int r = 0; r < nout; r++)
      s += dout[r] * w[r * nin + c];  // Transpose: use column c
    dx[c] += s;  // Accumulate (+=) because multiple paths may lead here
  }
}

/* Gradient w.r.t. weights: dW = dout @ x^T */
static inline void linear_bwd_w(const float *restrict x,
                                const float *restrict dout, int nout, int nin,
                                float *restrict dw) {
  for (int r = 0; r < nout; r++) {
    float dr = dout[r];
    float *dwr = dw + r * nin;
    for (int c = 0; c < nin; c++)
      dwr[c] += dr * x[c];  // Outer product
  }
}
```

**The calculus:**

Given `y = Wx` and loss L:
- ∂L/∂x = W^T (∂L/∂y)  —> chain rule with transposed W
- ∂L/∂W = (∂L/∂y) x^T  —> outer product

**Why accumulate `+=`?** Gradients from multiple operations sum together (chain rule).

### RMS Normalization Backward

This is the trickiest derivative. The forward was:
```
scale = 1 / √(mean(x²) + ε)
y = x * scale
```

The backward:

```c
static inline void rmsnorm_bwd(const float *x, float scale, const float *dout,
                               int n, float *dx) {
  // Compute dot product of gradient and normalized output
  float dot = 0;
  for (int i = 0; i < n; i++)
    dot += dout[i] * x[i];
  
  // The derivative has two terms:
  // 1. Direct term: scale * dout
  // 2. Correction term from the normalization itself
  float coeff = scale * scale * scale / n;
  
  for (int i = 0; i < n; i++)
    dx[i] += scale * dout[i] - coeff * x[i] * dot;
}
```

**The math:** This comes from the chain rule applied to RMSNorm. The second term accounts for how changing xᵢ affects the RMS of the entire vector, which then affects all outputs.

**Derivation sketch:**
```
∂L/∂xᵢ = ∂L/∂yⱼ ∂yⱼ/∂xᵢ = scale * ∂L/∂yᵢ - (scale³/n) xᵢ Σⱼ ∂L/∂yⱼ xⱼ
```

---

## Part 8: The Forward Pass - GPT in Action

Now we assemble everything into the forward pass. This is where the magic happens.

### Starting with Embeddings

```c
static void gpt_forward(int token_id, int pos_id, float *logits_out,
                        PosActs *act) {
  float x[N_EMBD], tmp[MLP_DIM > N_EMBD ? MLP_DIM : N_EMBD];
  
  // Step 1: Embed the token and add positional encoding
  for (int i = 0; i < N_EMBD; i++)
    x[i] = wte[token_id * N_EMBD + i] + wpe[pos_id * N_EMBD + i];
  
  memcpy(act->x_embed, x, sizeof(x));
```

**What's happening?**
- `wte[token_id]` looks up the token's learned embedding vector
- `wpe[pos_id]` adds position information (token 0 vs token 5 get different encodings)
- The sum becomes our initial representation

**Why add position?** Attention has no notion of order. Position embeddings inject "this is the 3rd token" information.

### Initial Normalization

```c
  // Step 2: Normalize before entering transformer layers
  act->rms_scale_init = rmsnorm_fwd(x, N_EMBD, x);
```

Modern transformers normalize inputs for stability. We normalize in-place (x is both input and output).

### The Transformer Layer

Now the heart of GPT the transformer block. We'll walk through one layer (we only have 1, but the code supports multiple):

```c
  for (int li = 0; li < N_LAYER; li++) {
    memcpy(act->x_in[li], x, sizeof(x));  // Save for residual connection
    
    // Step 3: Pre-attention normalization
    float xn[N_EMBD];
    act->rms_scale_attn[li] = rmsnorm_fwd(x, N_EMBD, xn);
    memcpy(act->xn_attn[li], xn, sizeof(xn));
```

**Residual connections:** We save `x` before modifying it. Later we'll add it back (skip connection).

### Multi-Head Self-Attention: Query, Key, Value Projections

```c
    // Step 4: Project to Query, Key, Value
    float q[N_EMBD], k[N_EMBD], v[N_EMBD];
    linear_fwd(xn, attn_wq[li], N_EMBD, N_EMBD, q);
    linear_fwd(xn, attn_wk[li], N_EMBD, N_EMBD, k);
    linear_fwd(xn, attn_wv[li], N_EMBD, N_EMBD, v);
    memcpy(act->q[li], q, sizeof(q));
    
    // Step 5: Store Key and Value in cache
    memcpy(kv_keys[li][pos_id], k, sizeof(k));
    memcpy(kv_vals[li][pos_id], v, sizeof(v));
    int seq_len = pos_id + 1;  // How many tokens we've seen
```

**The QKV transformation:**
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What information do I have?"
- **Value (V):** "What information should I send?"

Each token gets all three. Then we compute attention by comparing queries with keys.

### Computing Attention Weights

```c
    float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);
    float ao[N_EMBD];  // Attention output
    
    for (int h = 0; h < N_HEAD; h++) {
      int hs = h * HEAD_DIM;  // Head start index
      
      // Step 6: Compute attention logits (Q · K)
      float al[BLOCK_SIZE];  // Attention logits
      for (int tt = 0; tt < seq_len; tt++) {
        float dot = 0;
        for (int j = 0; j < HEAD_DIM; j++)
          dot += q[hs + j] * kv_keys[li][tt][hs + j];
        al[tt] = dot * scale;
      }
```

**Attention computation:**
```
attention_logits[t] = (Query · Key[t]) / √(head_dim)
```

**Why scale by √head_dim?** Without scaling, dot products grow with dimension, making softmax too sharp. Scaling keeps variance at 1.

### Softmax and Attention Output

```c
      // Step 7: Softmax to get attention weights
      float mx = al[0];
      for (int tt = 1; tt < seq_len; tt++)
        if (al[tt] > mx) mx = al[tt];
      
      float sm = 0;
      for (int tt = 0; tt < seq_len; tt++) {
        al[tt] = expf(al[tt] - mx);
        sm += al[tt];
      }
      
      float inv = 1.0f / sm;
      for (int tt = 0; tt < seq_len; tt++)
        al[tt] *= inv;
      
      // Save attention weights for backward pass
      for (int tt = 0; tt < seq_len; tt++)
        act->aw[li][h][tt] = al[tt];
```

Now `al[tt]` contains probabilities: "how much should I attend to position tt?"

### Weighted Sum of Values

```c
      // Step 8: Weighted sum of values
      for (int j = 0; j < HEAD_DIM; j++) {
        float s = 0;
        for (int tt = 0; tt < seq_len; tt++)
          s += al[tt] * kv_vals[li][tt][hs + j];
        ao[hs + j] = s;
      }
    }  // End of head loop
    
    memcpy(act->attn_out[li], ao, sizeof(ao));
```

**What just happened?** Each head computed:
```
output = Σ attention_weight[t] * Value[t]
```

This is a weighted average where weights are "how relevant is position t?"

### Output Projection and Residual Connection

```c
    // Step 9: Project attention output and add residual
    linear_fwd(ao, attn_wo[li], N_EMBD, N_EMBD, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + act->x_in[li][i];  // Residual connection!
    memcpy(act->x_mid[li], x, sizeof(x));
```

**Residual connection:** `x = f(x) + x` instead of `x = f(x)`. This creates gradient highways, making deep networks trainable.

### The MLP (Feedforward Network)

After attention comes a simple feedforward network:

```c
    // Step 10: Pre-MLP normalization
    float xn_m[N_EMBD];
    act->rms_scale_mlp[li] = rmsnorm_fwd(x, N_EMBD, xn_m);
    memcpy(act->xn_mlp[li], xn_m, sizeof(xn_m));
    
    // Step 11: First MLP layer
    float h1[MLP_DIM];
    linear_fwd(xn_m, mlp_fc1[li], MLP_DIM, N_EMBD, h1);
    memcpy(act->mlp_pre[li], h1, MLP_DIM * sizeof(float));
    
    // Step 12: Squared ReLU activation
    float h2[MLP_DIM];
    for (int i = 0; i < MLP_DIM; i++)
      h2[i] = h1[i] > 0 ? h1[i] * h1[i] : 0;
    memcpy(act->mlp_post[li], h2, MLP_DIM * sizeof(float));
```

**Squared ReLU:** Instead of `ReLU(x) = max(0, x)`, we use `x² if x > 0 else 0`. This is a newer variant that works well. It's smooth and has nice gradient properties.

### Final MLP Projection and Residual

```c
    // Step 13: Second MLP layer and residual
    linear_fwd(h2, mlp_fc2[li], N_EMBD, MLP_DIM, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + act->x_mid[li][i];  // Another residual!
  }  // End of layer loop
```

Now we've completed the transformer layer(s). The vector `x` contains the processed representation.

### Language Model Head

Finally, we project to vocabulary size to get logits for each possible next token:

```c
  // Step 14: Final projection to vocabulary
  memcpy(act->x_out, x, sizeof(x));
  linear_fwd(x, lm_head, vocab_size, N_EMBD, logits_out);
}
```

**Logits:** These are un-normalized scores. High logit = model thinks this token is likely next.

---

## Part 9: The Backward Pass - Backpropagation

This is where we compute gradients. We walk backward through the computation graph, accumulating ∂Loss/∂parameter for every parameter.

### Overview and Setup

```c
static void gpt_backward(int n, const int *tokens, const int *targets) {
  memset(dk_accum, 0, sizeof(dk_accum));  // Clear KV gradient accumulators
  memset(dv_accum, 0, sizeof(dv_accum));
  float inv_n = 1.0f / n;  // Average gradient over sequence
```

We process positions in reverse order (backward pass goes... backward).

### Starting from the Loss

```c
  for (int pos = n - 1; pos >= 0; pos--) {
    PosActs *act = &saved[pos];
    int seq_len = pos + 1;
    
    // Step 1: Gradient of cross-entropy loss
    float dl[MAX_CHARS + 1];
    for (int i = 0; i < vocab_size; i++)
      dl[i] = (saved_probs[pos][i] - (i == targets[pos] ? 1.0f : 0.0f)) * inv_n;
```

**Cross-entropy derivative:** For softmax + cross-entropy, the gradient is simply:
```
∂L/∂logit[i] = probability[i] - 1   if i is the target
              = probability[i]       otherwise
```

This is a beautiful simplification! The derivative of `softmax(x) with cross_entropy` is just `softmax(x) - target_one_hot`.

### Language Model Head Backward

```c
    // Step 2: Backprop through language model head
    float dx[N_EMBD];
    memset(dx, 0, sizeof(dx));
    linear_bwd_x(lm_head, dl, vocab_size, N_EMBD, dx);      // Gradient to x
    linear_bwd_w(act->x_out, dl, vocab_size, N_EMBD, d_lm_head);  // Gradient to weights
```

We compute:
- `dx`: how changing the final hidden state affects loss
- `d_lm_head`: how changing language model weights affects loss

### MLP Backward

Now we backprop through each layer (in reverse):

```c
    for (int li = N_LAYER - 1; li >= 0; li--) {
      /* === MLP Backward === */
      
      // Step 3: Backprop through second MLP layer
      float d_h2[MLP_DIM];
      memset(d_h2, 0, sizeof(d_h2));
      linear_bwd_x(mlp_fc2[li], dx, N_EMBD, MLP_DIM, d_h2);
      linear_bwd_w(act->mlp_post[li], dx, N_EMBD, MLP_DIM, d_mlp_fc2[li]);
      
      // Step 4: Backprop through squared ReLU
      float d_h1[MLP_DIM];
      for (int i = 0; i < MLP_DIM; i++)
        d_h1[i] = act->mlp_pre[li][i] > 0 
                  ? 2.0f * act->mlp_pre[li][i] * d_h2[i]  // d/dx(x²) = 2x
                  : 0;
```

**Squared ReLU derivative:**
```
if x > 0: d/dx(x²) = 2x
if x ≤ 0: d/dx(0) = 0
```

We multiply by the gradient from above (`d_h2[i]`) via chain rule.

```c
      // Step 5: Backprop through first MLP layer
      float d_xn_mlp[N_EMBD];
      memset(d_xn_mlp, 0, sizeof(d_xn_mlp));
      linear_bwd_x(mlp_fc1[li], d_h1, MLP_DIM, N_EMBD, d_xn_mlp);
      linear_bwd_w(act->xn_mlp[li], d_h1, MLP_DIM, N_EMBD, d_mlp_fc1[li]);
      
      // Step 6: Backprop through MLP normalization
      float d_x_mid[N_EMBD];
      memset(d_x_mid, 0, sizeof(d_x_mid));
      rmsnorm_bwd(act->x_mid[li], act->rms_scale_mlp[li], d_xn_mlp, N_EMBD, d_x_mid);
      
      // Step 7: Add gradient from residual connection
      for (int i = 0; i < N_EMBD; i++)
        dx[i] += d_x_mid[i];  // Residual splits gradient
```

**Residual connection gradient:** When forward had `y = f(x) + x`, backward has `dx += dy`. The gradient flows through both the function AND the skip connection.

### Attention Backward (The Hard Part)

Attention backward is complex because attention couples all positions together:

```c
      /* === Attention Backward === */
      
      // Step 8: Backprop through attention output projection
      float d_ao[N_EMBD];
      memset(d_ao, 0, sizeof(d_ao));
      linear_bwd_x(attn_wo[li], dx, N_EMBD, N_EMBD, d_ao);
      linear_bwd_w(act->attn_out[li], dx, N_EMBD, N_EMBD, d_attn_wo[li]);
      
      float d_q[N_EMBD];
      memset(d_q, 0, sizeof(d_q));
      float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);
      
      for (int h = 0; h < N_HEAD; h++) {
        int hs = h * HEAD_DIM;
        
        // Step 9: Gradient w.r.t. attention output for this head
        float d_aw[BLOCK_SIZE];  // Gradient w.r.t. attention weights
        memset(d_aw, 0, sizeof(d_aw));
        
        for (int j = 0; j < HEAD_DIM; j++) {
          for (int tt = 0; tt < seq_len; tt++) {
            // Backprop through: output[j] = Σ attention_weight[tt] * value[tt][j]
            d_aw[tt] += d_ao[hs + j] * kv_vals[li][tt][hs + j];
            dv_accum[li][tt][hs + j] += act->aw[li][h][tt] * d_ao[hs + j];
          }
        }
```

**What's happening?** The forward was:
```
output[j] = Σ_t attention_weight[t] * value[t][j]
```

Backward:
```
d_attention_weight[t] += d_output[j] * value[t][j]  (for all j)
d_value[t][j] += attention_weight[t] * d_output[j]
```

### Softmax Backward

```c
        // Step 10: Backprop through softmax
        float dot = 0;
        for (int tt = 0; tt < seq_len; tt++)
          dot += d_aw[tt] * act->aw[li][h][tt];
        
        float d_al[BLOCK_SIZE];  // Gradient w.r.t. attention logits
        for (int tt = 0; tt < seq_len; tt++)
          d_al[tt] = act->aw[li][h][tt] * (d_aw[tt] - dot);
```

**Softmax Jacobian:** The derivative of softmax is:
```
∂softmax[i]/∂logit[j] = softmax[i] * (δᵢⱼ - softmax[j])
```

Where δᵢⱼ is 1 if i=j, else 0. This leads to:
```
d_logit[j] = softmax[j] * (d_softmax[j] - Σᵢ d_softmax[i] * softmax[i])
```

### Query and Key Gradients

```c
        // Step 11: Backprop through scaled dot product attention
        for (int tt = 0; tt < seq_len; tt++) {
          for (int j = 0; j < HEAD_DIM; j++) {
            // Backprop through: logit[tt] = (query[j] * key[tt][j]) * scale
            d_q[hs + j] += d_al[tt] * kv_keys[li][tt][hs + j] * scale;
            dk_accum[li][tt][hs + j] += d_al[tt] * act->q[li][hs + j] * scale;
          }
        }
      }  // End of head loop
```

**Dot product derivative:** If `z = x · y`, then:
```
∂z/∂x = y
∂z/∂y = x
```

Plus we multiply by `scale` since forward did.

### QKV Projections Backward

```c
      // Step 12: Backprop through Q, K, V projections
      float d_xn[N_EMBD];
      memset(d_xn, 0, sizeof(d_xn));
      
      // Query gradient
      linear_bwd_x(attn_wq[li], d_q, N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], d_q, N_EMBD, N_EMBD, d_attn_wq[li]);
      
      // Key gradient
      linear_bwd_x(attn_wk[li], dk_accum[li][pos], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], dk_accum[li][pos], N_EMBD, N_EMBD, d_attn_wk[li]);
      
      // Value gradient
      linear_bwd_x(attn_wv[li], dv_accum[li][pos], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], dv_accum[li][pos], N_EMBD, N_EMBD, d_attn_wv[li]);
```

All three projections contribute to the gradient of the normalized input `d_xn`.

### Completing the Layer

```c
      // Step 13: Backprop through attention normalization
      float d_x_in[N_EMBD];
      memset(d_x_in, 0, sizeof(d_x_in));
      rmsnorm_bwd(act->x_in[li], act->rms_scale_attn[li], d_xn, N_EMBD, d_x_in);
      
      // Step 14: Add gradient from residual connection
      for (int i = 0; i < N_EMBD; i++)
        dx[i] = dx[i] + d_x_in[i];
    }  // End of layer loop
```

### Embedding Gradients

Finally, gradients flow back to the embeddings:

```c
    // Step 15: Backprop through initial normalization
    float d_embed[N_EMBD];
    memset(d_embed, 0, sizeof(d_embed));
    rmsnorm_bwd(act->x_embed, act->rms_scale_init, dx, N_EMBD, d_embed);
    
    // Step 16: Accumulate gradients for token and position embeddings
    int tok = tokens[pos];
    for (int i = 0; i < N_EMBD; i++) {
      d_wte[tok * N_EMBD + i] += d_embed[i];
      d_wpe[pos * N_EMBD + i] += d_embed[i];
    }
  }  // End of position loop
}
```

**Why accumulate?** Multiple positions might use the same token, so we add all their gradients together.

---

## Part 10: Optimization - Adam Optimizer

Now we have gradients. We need to update parameters. We'll use **Adam** (Adaptive Moment Estimation), which is much better than plain SGD.

### Adam Update Step

```c
static void adam_update(float *p, float *g, float *m, float *v, int sz,
                        float lr, float b1, float b2, float eps, int step) {
  // Bias correction factors
  float b1c = 1.0f - powf(b1, step + 1);
  float b2c = 1.0f - powf(b2, step + 1);
  
  for (int i = 0; i < sz; i++) {
    // Update first moment (momentum)
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    
    // Update second moment (variance)
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    
    // Bias-corrected update
    p[i] -= lr * (m[i] / b1c) / (sqrtf(v[i] / b2c) + eps);
    
    // Clear gradient for next iteration
    g[i] = 0;
  }
}
```

**Adam intuition:**

- **m (momentum):** Exponential moving average of gradients. Helps accelerate in consistent directions.
- **v (variance):** Exponential moving average of squared gradients. Adapts learning rate per-parameter.
- **Bias correction:** Early in training, m and v are biased toward 0. We correct for this.

**The update rule:**
```
m = β₁ * m + (1 - β₁) * gradient
v = β₂ * v + (1 - β₂) * gradient²
parameter -= learning_rate * (m / √v)
```

**Why it works:** Parameters with large, consistent gradients get bigger updates. Parameters with noisy gradients get smaller updates (because √v is larger).

---

## Part 11: Sampling from the Model

To generate text, we sample from the probability distribution:

```c
static int weighted_choice(const float *w, int n) {
  // Sum all probabilities
  float total = 0;
  for (int i = 0; i < n; i++)
    total += w[i];
  
  // Pick a random point in [0, total)
  float r = (float)rng_uniform() * total;
  float cum = 0;
  
  // Find which token that point lands on
  for (int i = 0; i < n; i++) {
    cum += w[i];
    if (r < cum)
      return i;
  }
  
  return n - 1;
}
```

**How it works:** Imagine laying all probabilities end-to-end on a line. Throw a dart randomly. Whichever probability segment it lands in, return that token.

**Example:** If probabilities are [0.7, 0.2, 0.1]:
- 0 to 0.7: return token 0
- 0.7 to 0.9: return token 1
- 0.9 to 1.0: return token 2

---

## Part 12: Training Loop - Putting It All Together

Now we wire everything together into a training loop:

```c
int main(void) {
  // Step 1: Load data and prepare
  load_dataset("input.txt");
  
  // Shuffle documents for randomness
  int *doc_order = (int *)malloc(num_docs * sizeof(int));
  for (int i = 0; i < num_docs; i++)
    doc_order[i] = i;
  shuffle_ints(doc_order, num_docs);
  
  char (*docs_tmp)[MAX_DOC_LEN] = malloc((size_t)num_docs * MAX_DOC_LEN);
  for (int i = 0; i < num_docs; i++)
    memcpy(docs_tmp[i], docs[doc_order[i]], MAX_DOC_LEN);
  memcpy(docs, docs_tmp, (size_t)num_docs * MAX_DOC_LEN);
  free(docs_tmp);
  free(doc_order);
  
  printf("num docs: %d\n", num_docs);
  
  // Step 2: Build tokenizer and initialize model
  build_tokenizer();
  printf("vocab size: %d\n", vocab_size);
  init_params();
```

**Why shuffle?** We want the model to see documents in random order, not always in the same sequence. This prevents overfitting to ordering.

### Training Hyperparameters

```c
  // Step 3: Set hyperparameters
  float lr = 1e-3f;      // Learning rate
  float b1 = 0.9f;       // Adam beta1 (momentum)
  float b2 = 0.95f;      // Adam beta2 (variance)
  float eps = 1e-8f;     // Adam epsilon (numerical stability)
  int num_steps = 5000;   // Training steps
```

These are typical values for small models. Larger models often use smaller learning rates (1e-4 to 1e-5).

### Training Step

```c
  for (int step = 0; step < num_steps; step++) {
    // Get current document (cycle through dataset)
    char *doc = docs[step % num_docs];
    int doc_len = (int)strlen(doc);
    
    // Tokenize: [BOS, char1, char2, ..., charN, BOS]
    int tokens[MAX_DOC_LEN + 2], targets[BLOCK_SIZE];
    tokens[0] = BOS;
    for (int i = 0; i < doc_len; i++)
      tokens[i + 1] = char_to_id(doc[i]);
    tokens[doc_len + 1] = BOS;
    
    // Limit sequence length to BLOCK_SIZE
    int n = BLOCK_SIZE < (doc_len + 1) ? BLOCK_SIZE : (doc_len + 1);
```

**Tokenization:** We wrap text in BOS tokens. This gives the model clear start/end boundaries.

### Forward Pass

```c
    // Forward pass and loss computation
    float total_loss = 0;
    float logits[MAX_CHARS + 1];
    
    for (int pos = 0; pos < n; pos++) {
      targets[pos] = tokens[pos + 1];  // Predict next token
      
      // Forward pass for this position
      gpt_forward(tokens[pos], pos, logits, &saved[pos]);
      
      // Convert logits to probabilities
      softmax_fwd(logits, vocab_size, saved_probs[pos]);
      
      // Compute cross-entropy loss
      total_loss += -logf(saved_probs[pos][targets[pos]] + 1e-30f);
    }
    
    float loss = total_loss / n;
```

**Cross-entropy loss:** For each position, we compute:
```
loss = -log(probability_of_correct_token)
```

Good predictions (high probability) give low loss. Bad predictions give high loss.

**Why 1e-30?** Prevents log(0) which is undefined. If probability is exactly 0 (shouldn't happen), we use a tiny value.

### Backward Pass and Optimization

```c
    // Backward pass
    gpt_backward(n, tokens, targets);
    
    // Cosine learning rate schedule
    float lr_t = lr * 0.5f * (1.0f + cosf((float)M_PI * step / (float)num_steps));
```

**Cosine schedule:** Learning rate starts at `lr`, decreases following a cosine curve to ~0 at the end. This helps the model settle into a good minimum.

```c
    // Update all parameters with Adam
    int es = vocab_size * N_EMBD;
    int ps = BLOCK_SIZE * N_EMBD;
    int as = N_EMBD * N_EMBD;
    int ms = MLP_DIM * N_EMBD;
    
    adam_update(wte, d_wte, adam_m_wte, adam_v_wte, es, lr_t, b1, b2, eps, step);
    adam_update(wpe, d_wpe, adam_m_wpe, adam_v_wpe, ps, lr_t, b1, b2, eps, step);
    adam_update(lm_head, d_lm_head, adam_m_lm, adam_v_lm, es, lr_t, b1, b2, eps, step);
    
    for (int i = 0; i < N_LAYER; i++) {
      adam_update(attn_wq[i], d_attn_wq[i], adam_m_wq[i], adam_v_wq[i], as, lr_t, b1, b2, eps, step);
      adam_update(attn_wk[i], d_attn_wk[i], adam_m_wk[i], adam_v_wk[i], as, lr_t, b1, b2, eps, step);
      adam_update(attn_wv[i], d_attn_wv[i], adam_m_wv[i], adam_v_wv[i], as, lr_t, b1, b2, eps, step);
      adam_update(attn_wo[i], d_attn_wo[i], adam_m_wo[i], adam_v_wo[i], as, lr_t, b1, b2, eps, step);
      adam_update(mlp_fc1[i], d_mlp_fc1[i], adam_m_fc1[i], adam_v_fc1[i], ms, lr_t, b1, b2, eps, step);
      adam_update(mlp_fc2[i], d_mlp_fc2[i], adam_m_fc2[i], adam_v_fc2[i], ms, lr_t, b1, b2, eps, step);
    }
    
    printf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss);
  }
```

We update every parameter group. Watch the loss decrease!

---

## Part 13: Inference - Generating Text

After training, we sample from the model:

```c
  /* === Inference === */
  float temperature = 0.5f;
  printf("\n--- inference ---\n");
  
  for (int si = 0; si < 20; si++) {
    char sample[BLOCK_SIZE + 1];
    int slen = 0, token_id = BOS;
    PosActs tmp_act;
    
    // Generate one sequence
    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
      float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
      
      // Forward pass
      gpt_forward(token_id, pos, logits, &tmp_act);
      
      // Apply temperature
      float inv_t = 1.0f / temperature;
      for (int i = 0; i < vocab_size; i++)
        logits[i] *= inv_t;
      
      // Sample from distribution
      softmax_fwd(logits, vocab_size, probs);
      token_id = weighted_choice(probs, vocab_size);
      
      // Stop at BOS (end of sequence)
      if (token_id == BOS)
        break;
      
      // Add character to sample
      if (token_id < num_uchars)
        sample[slen++] = uchars_arr[token_id];
    }
    
    sample[slen] = '\0';
    printf("sample %2d: %s\n", si + 1, sample);
    
    // Clear KV cache for next sample
    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));
  }
```

**Temperature:** Controls randomness:
- `temperature = 1.0`: Sample exactly from model distribution
- `temperature < 1.0`: More confident (sharper distribution)
- `temperature > 1.0`: More random (flatter distribution)

**How it works:** Dividing logits by temperature before softmax changes the distribution shape.

### Cleanup

```c
  /* Free all allocated memory */
  free(wte);
  free(d_wte);
  free(adam_m_wte);
  free(adam_v_wte);
  // ... (free all other parameters)
  
  return 0;
}
```

Always free your malloc'd memory!

---

## Compilation and Running

Compile with optimizations:

```bash
gcc -O3 -march=native -ffast-math -o gpt gpt.c -lm
```

**Flags explained:**
- `-O3`: Maximum optimization
- `-march=native`: Use CPU-specific instructions (AVX, etc.)
- `-ffast-math`: Faster floating point (trades some precision)
- `-lm`: Link math library (for sqrt, exp, etc.)

Create an `input.txt` with training data (one document per line):

```
can you make those wheels spin around?
that's pretty cool.
look at that.
vroom vroom vroom.
bang bang.
.
.
.
<80000+ lines>
```

Run:

```bash
./gpt
```

You should see loss decreasing, then generated samples!
