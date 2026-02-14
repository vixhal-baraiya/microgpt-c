/*
 * Enhanced GPT implementation in pure C with extended features:
 * - Model checkpointing (save/load weights)
 * - Train/validation split with tracking
 * - Top-k and top-p (nucleus) sampling
 * - Gradient clipping for stability
 * - Learning rate warmup
 * - Running loss statistics and best model tracking
 * - Configurable sampling strategies
 * - Better progress reporting (tokens/sec, ETA)
 *
 * Compile: gcc -O3 -march=native -ffast-math -o gpt gpt_improved.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/*  Minimal xorshift PRNG (seeded deterministically like Python's 42) */
/* ------------------------------------------------------------------ */
static unsigned long long rng_state = 42;

static unsigned long long rng_next(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

static double rng_uniform(void) {
  return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static float rng_gauss(float mean, float std) {
  double u1 = rng_uniform(), u2 = rng_uniform();
  if (u1 < 1e-30)
    u1 = 1e-30;
  return mean + std * (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

static void shuffle_ints(int *arr, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(rng_uniform() * (i + 1));
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

/* ------------------------------------------------------------------ */
/*  Dataset loading with train/val split                             */
/* ------------------------------------------------------------------ */
#define MAX_DOCS 85000
#define MAX_DOC_LEN 512
#define MAX_CHARS 128

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int num_docs = 0;
static int num_train_docs = 0;
static int num_val_docs = 0;

static void load_dataset(const char *filename, float val_split) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", filename);
    exit(1);
  }
  char line[256];
  while (fgets(line, sizeof(line), f) && num_docs < MAX_DOCS) {
    int len = (int)strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = 0;
    if (len > 0) {
      strncpy(docs[num_docs], line, MAX_DOC_LEN - 1);
      docs[num_docs][MAX_DOC_LEN - 1] = 0;
      num_docs++;
    }
  }
  fclose(f);
  
  num_val_docs = (int)(num_docs * val_split);
  num_train_docs = num_docs - num_val_docs;
}

/* ------------------------------------------------------------------ */
/*  Tokenizer                                                         */
/* ------------------------------------------------------------------ */
static char uchars_arr[MAX_CHARS];
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
  int seen[256] = {0};
  for (int d = 0; d < num_docs; d++)
    for (int i = 0; docs[d][i]; i++)
      seen[(unsigned char)docs[d][i]] = 1;
  for (int i = 0; i < 256; i++)
    if (seen[i])
      uchars_arr[num_uchars++] = (char)i;
  qsort(uchars_arr, num_uchars, sizeof(char), cmp_char);
  BOS = num_uchars;
  vocab_size = num_uchars + 1;
}

/* ------------------------------------------------------------------ */
/*  Model hyper-parameters                                            */
/* ------------------------------------------------------------------ */
#define N_EMBD 32
#define N_HEAD 4
#define N_LAYER 1
#define BLOCK_SIZE 8
#define HEAD_DIM (N_EMBD / N_HEAD)
#define MLP_DIM (4 * N_EMBD)

/* ------------------------------------------------------------------ */
/*  Parameters & gradients (float arrays)                             */
/* ------------------------------------------------------------------ */
static float *wte, *d_wte;
static float *wpe, *d_wpe;
static float *lm_head, *d_lm_head;

static float *attn_wq[N_LAYER], *d_attn_wq[N_LAYER];
static float *attn_wk[N_LAYER], *d_attn_wk[N_LAYER];
static float *attn_wv[N_LAYER], *d_attn_wv[N_LAYER];
static float *attn_wo[N_LAYER], *d_attn_wo[N_LAYER];
static float *mlp_fc1[N_LAYER], *d_mlp_fc1[N_LAYER];
static float *mlp_fc2[N_LAYER], *d_mlp_fc2[N_LAYER];

/* Adam optimizer buffers */
static float *adam_m_wte, *adam_v_wte;
static float *adam_m_wpe, *adam_v_wpe;
static float *adam_m_lm, *adam_v_lm;
static float *adam_m_wq[N_LAYER], *adam_v_wq[N_LAYER];
static float *adam_m_wk[N_LAYER], *adam_v_wk[N_LAYER];
static float *adam_m_wv[N_LAYER], *adam_v_wv[N_LAYER];
static float *adam_m_wo[N_LAYER], *adam_v_wo[N_LAYER];
static float *adam_m_fc1[N_LAYER], *adam_v_fc1[N_LAYER];
static float *adam_m_fc2[N_LAYER], *adam_v_fc2[N_LAYER];

static int num_params = 0;

static float *make_param(int size, float std) {
  float *p = (float *)calloc(size, sizeof(float));
  for (int i = 0; i < size; i++)
    p[i] = rng_gauss(0, std);
  num_params += size;
  return p;
}

static float *make_zero(int size) {
  return (float *)calloc(size, sizeof(float));
}

static void init_params(void) {
  int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
  int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
  wte = make_param(es, 0.02f);
  d_wte = make_zero(es);
  adam_m_wte = make_zero(es);
  adam_v_wte = make_zero(es);
  wpe = make_param(ps, 0.02f);
  d_wpe = make_zero(ps);
  adam_m_wpe = make_zero(ps);
  adam_v_wpe = make_zero(ps);
  lm_head = make_param(es, 0.02f);
  d_lm_head = make_zero(es);
  adam_m_lm = make_zero(es);
  adam_v_lm = make_zero(es);
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

/* ------------------------------------------------------------------ */
/*  Model checkpointing: save/load weights                            */
/* ------------------------------------------------------------------ */
static void save_checkpoint(const char *filename, int step, float best_loss) {
  FILE *f = fopen(filename, "wb");
  if (!f) {
    fprintf(stderr, "Cannot save checkpoint to %s\n", filename);
    return;
  }
  
  fwrite(&step, sizeof(int), 1, f);
  fwrite(&best_loss, sizeof(float), 1, f);
  fwrite(&vocab_size, sizeof(int), 1, f);
  
  int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
  int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
  
  fwrite(wte, sizeof(float), es, f);
  fwrite(wpe, sizeof(float), ps, f);
  fwrite(lm_head, sizeof(float), es, f);
  
  for (int i = 0; i < N_LAYER; i++) {
    fwrite(attn_wq[i], sizeof(float), as, f);
    fwrite(attn_wk[i], sizeof(float), as, f);
    fwrite(attn_wv[i], sizeof(float), as, f);
    fwrite(attn_wo[i], sizeof(float), as, f);
    fwrite(mlp_fc1[i], sizeof(float), ms, f);
    fwrite(mlp_fc2[i], sizeof(float), ms, f);
  }
  
  fclose(f);
  printf("Checkpoint saved to %s (step %d, loss %.4f)\n", filename, step, best_loss);
}

static int load_checkpoint(const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    return 0;
  }
  
  int step;
  float best_loss;
  int saved_vocab_size;
  
  fread(&step, sizeof(int), 1, f);
  fread(&best_loss, sizeof(float), 1, f);
  fread(&saved_vocab_size, sizeof(int), 1, f);
  
  if (saved_vocab_size != vocab_size) {
    fprintf(stderr, "Vocab size mismatch: saved=%d, current=%d\n", 
            saved_vocab_size, vocab_size);
    fclose(f);
    return 0;
  }
  
  int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
  int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
  
  fread(wte, sizeof(float), es, f);
  fread(wpe, sizeof(float), ps, f);
  fread(lm_head, sizeof(float), es, f);
  
  for (int i = 0; i < N_LAYER; i++) {
    fread(attn_wq[i], sizeof(float), as, f);
    fread(attn_wk[i], sizeof(float), as, f);
    fread(attn_wv[i], sizeof(float), as, f);
    fread(attn_wo[i], sizeof(float), as, f);
    fread(mlp_fc1[i], sizeof(float), ms, f);
    fread(mlp_fc2[i], sizeof(float), ms, f);
  }
  
  fclose(f);
  printf("Checkpoint loaded from %s (step %d, loss %.4f)\n", filename, step, best_loss);
  return step;
}

/* ------------------------------------------------------------------ */
/*  Saved activations for backward pass                               */
/* ------------------------------------------------------------------ */
typedef struct {
  float x_embed[N_EMBD];
  float rms_scale_init;
  float x_in[N_LAYER][N_EMBD];
  float xn_attn[N_LAYER][N_EMBD];
  float rms_scale_attn[N_LAYER];
  float q[N_LAYER][N_EMBD];
  float aw[N_LAYER][N_HEAD][BLOCK_SIZE];
  float attn_out[N_LAYER][N_EMBD];
  float x_mid[N_LAYER][N_EMBD];
  float xn_mlp[N_LAYER][N_EMBD];
  float rms_scale_mlp[N_LAYER];
  float mlp_pre[N_LAYER][MLP_DIM];
  float mlp_post[N_LAYER][MLP_DIM];
  float x_out[N_EMBD];
} PosActs;

static PosActs saved[BLOCK_SIZE];
static float saved_probs[BLOCK_SIZE][MAX_CHARS + 1];

/* KV cache & gradient accumulators */
static float kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static float kv_vals[N_LAYER][BLOCK_SIZE][N_EMBD];
static float dk_accum[N_LAYER][BLOCK_SIZE][N_EMBD];
static float dv_accum[N_LAYER][BLOCK_SIZE][N_EMBD];

/* ------------------------------------------------------------------ */
/*  Forward building blocks (inlined for speed)                       */
/* ------------------------------------------------------------------ */
static inline void linear_fwd(const float *restrict x, const float *restrict w,
                              int nout, int nin, float *restrict out) {
  for (int r = 0; r < nout; r++) {
    float s = 0;
    for (int c = 0; c < nin; c++)
      s += x[c] * w[r * nin + c];
    out[r] = s;
  }
}

static inline void linear_bwd_x(const float *restrict w, const float *restrict dout,
                                int nout, int nin, float *restrict dx) {
  for (int c = 0; c < nin; c++)
    for (int r = 0; r < nout; r++)
      dx[c] += dout[r] * w[r * nin + c];
}

static inline void linear_bwd_w(const float *restrict x, const float *restrict dout,
                                int nout, int nin, float *restrict dw) {
  for (int r = 0; r < nout; r++)
    for (int c = 0; c < nin; c++)
      dw[r * nin + c] += dout[r] * x[c];
}

static inline void rmsnorm_fwd(const float *x, int n, float *out,
                               float *scale_out) {
  float ss = 0;
  for (int i = 0; i < n; i++)
    ss += x[i] * x[i];
  float scale = 1.0f / sqrtf(ss / n + 1e-5f);
  *scale_out = scale;
  for (int i = 0; i < n; i++)
    out[i] = x[i] * scale;
}

static inline void rmsnorm_bwd(const float *x, float scale, const float *dout,
                               int n, float *dx) {
  float dot = 0, ss = 0;
  for (int i = 0; i < n; i++) {
    dot += dout[i] * x[i];
    ss += x[i] * x[i];
  }
  float inv_rms = sqrtf(n / (ss + 1e-5f * n));
  float coef = -dot * inv_rms * inv_rms * inv_rms / n;
  for (int i = 0; i < n; i++)
    dx[i] = dout[i] * scale + coef * x[i];
}

static inline void gelu_fwd(const float *x, int n, float *out) {
  for (int i = 0; i < n; i++) {
    float c = 0.044715f * x[i] * x[i] * x[i];
    float t = tanhf(0.7978845608f * (x[i] + c));
    out[i] = 0.5f * x[i] * (1.0f + t);
  }
}

static inline void gelu_bwd(const float *x, const float *dout, int n,
                            float *dx) {
  for (int i = 0; i < n; i++) {
    float c = 0.044715f * x[i] * x[i] * x[i];
    float t = tanhf(0.7978845608f * (x[i] + c));
    float dt = (1.0f - t * t) * 0.7978845608f *
               (1.0f + 3.0f * 0.044715f * x[i] * x[i]);
    dx[i] += dout[i] * (0.5f * (1.0f + t) + 0.5f * x[i] * dt);
  }
}

static inline void softmax_fwd(const float *logits, int n, float *out) {
  float m = logits[0];
  for (int i = 1; i < n; i++)
    if (logits[i] > m)
      m = logits[i];
  float s = 0;
  for (int i = 0; i < n; i++) {
    out[i] = expf(logits[i] - m);
    s += out[i];
  }
  for (int i = 0; i < n; i++)
    out[i] /= s;
}

/* ------------------------------------------------------------------ */
/*  GPT forward pass (single token)                                   */
/* ------------------------------------------------------------------ */
static void gpt_forward(int token_id, int pos, float *logits, PosActs *act) {
  for (int i = 0; i < N_EMBD; i++)
    act->x_embed[i] = wte[token_id * N_EMBD + i] + wpe[pos * N_EMBD + i];

  float xn[N_EMBD];
  rmsnorm_fwd(act->x_embed, N_EMBD, xn, &act->rms_scale_init);

  for (int li = 0; li < N_LAYER; li++) {
    memcpy(act->x_in[li], li == 0 ? xn : act->x_out, N_EMBD * sizeof(float));
    rmsnorm_fwd(act->x_in[li], N_EMBD, act->xn_attn[li],
                &act->rms_scale_attn[li]);

    linear_fwd(act->xn_attn[li], attn_wq[li], N_EMBD, N_EMBD, act->q[li]);
    linear_fwd(act->xn_attn[li], attn_wk[li], N_EMBD, N_EMBD,
               kv_keys[li][pos]);
    linear_fwd(act->xn_attn[li], attn_wv[li], N_EMBD, N_EMBD,
               kv_vals[li][pos]);

    int seq_len = pos + 1;
    float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);
    for (int h = 0; h < N_HEAD; h++) {
      int hs = h * HEAD_DIM;
      float scores[BLOCK_SIZE], att[BLOCK_SIZE];
      for (int tt = 0; tt < seq_len; tt++) {
        float s = 0;
        for (int d = 0; d < HEAD_DIM; d++)
          s += act->q[li][hs + d] * kv_keys[li][tt][hs + d];
        scores[tt] = s * scale;
      }
      softmax_fwd(scores, seq_len, att);
      for (int tt = 0; tt < seq_len; tt++)
        act->aw[li][h][tt] = att[tt];
      for (int d = 0; d < HEAD_DIM; d++) {
        float o = 0;
        for (int tt = 0; tt < seq_len; tt++)
          o += att[tt] * kv_vals[li][tt][hs + d];
        act->attn_out[li][hs + d] = o;
      }
    }

    linear_fwd(act->attn_out[li], attn_wo[li], N_EMBD, N_EMBD, xn);
    for (int i = 0; i < N_EMBD; i++)
      act->x_mid[li][i] = act->x_in[li][i] + xn[i];

    rmsnorm_fwd(act->x_mid[li], N_EMBD, act->xn_mlp[li],
                &act->rms_scale_mlp[li]);
    linear_fwd(act->xn_mlp[li], mlp_fc1[li], MLP_DIM, N_EMBD, act->mlp_pre[li]);
    gelu_fwd(act->mlp_pre[li], MLP_DIM, act->mlp_post[li]);
    linear_fwd(act->mlp_post[li], mlp_fc2[li], N_EMBD, MLP_DIM, xn);
    for (int i = 0; i < N_EMBD; i++)
      act->x_out[i] = act->x_mid[li][i] + xn[i];
  }

  linear_fwd(act->x_out, lm_head, vocab_size, N_EMBD, logits);
}

/* ------------------------------------------------------------------ */
/*  GPT backward pass (full sequence)                                 */
/* ------------------------------------------------------------------ */
static void gpt_backward(int seq_len, const int *tokens, const int *targets) {
  memset(dk_accum, 0, sizeof(dk_accum));
  memset(dv_accum, 0, sizeof(dv_accum));

  for (int pos = seq_len - 1; pos >= 0; pos--) {
    PosActs *act = &saved[pos];
    float dx[N_EMBD];
    memset(dx, 0, sizeof(dx));

    for (int i = 0; i < vocab_size; i++) {
      float err = saved_probs[pos][i];
      if (i == targets[pos])
        err -= 1.0f;
      for (int j = 0; j < N_EMBD; j++) {
        dx[j] += err * lm_head[i * N_EMBD + j];
        d_lm_head[i * N_EMBD + j] += err * act->x_out[j];
      }
    }

    for (int li = N_LAYER - 1; li >= 0; li--) {
      float d_xn[N_EMBD];
      memset(d_xn, 0, sizeof(d_xn));
      rmsnorm_bwd(act->x_mid[li], act->rms_scale_mlp[li], dx, N_EMBD, d_xn);

      float d_mlp_post[MLP_DIM];
      memset(d_mlp_post, 0, sizeof(d_mlp_post));
      linear_bwd_x(mlp_fc2[li], d_xn, N_EMBD, MLP_DIM, d_mlp_post);
      linear_bwd_w(act->mlp_post[li], d_xn, N_EMBD, MLP_DIM, d_mlp_fc2[li]);

      float d_mlp_pre[MLP_DIM];
      memset(d_mlp_pre, 0, sizeof(d_mlp_pre));
      gelu_bwd(act->mlp_pre[li], d_mlp_post, MLP_DIM, d_mlp_pre);

      memset(d_xn, 0, sizeof(d_xn));
      linear_bwd_x(mlp_fc1[li], d_mlp_pre, MLP_DIM, N_EMBD, d_xn);
      linear_bwd_w(act->xn_mlp[li], d_mlp_pre, MLP_DIM, N_EMBD, d_mlp_fc1[li]);

      float d_x_mid[N_EMBD];
      memset(d_x_mid, 0, sizeof(d_x_mid));
      rmsnorm_bwd(act->x_mid[li], act->rms_scale_mlp[li], d_xn, N_EMBD,
                  d_x_mid);
      for (int i = 0; i < N_EMBD; i++)
        dx[i] = dx[i] + d_x_mid[i];

      float d_ao[N_EMBD];
      memset(d_ao, 0, sizeof(d_ao));
      linear_bwd_x(attn_wo[li], dx, N_EMBD, N_EMBD, d_ao);
      linear_bwd_w(act->attn_out[li], dx, N_EMBD, N_EMBD, d_attn_wo[li]);

      float d_q[N_EMBD];
      memset(d_q, 0, sizeof(d_q));
      float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);

      for (int h = 0; h < N_HEAD; h++) {
        int hs = h * HEAD_DIM;
        float d_aw[BLOCK_SIZE];
        memset(d_aw, 0, sizeof(d_aw));
        for (int j = 0; j < HEAD_DIM; j++) {
          for (int tt = 0; tt < seq_len; tt++) {
            d_aw[tt] += d_ao[hs + j] * kv_vals[li][tt][hs + j];
            dv_accum[li][tt][hs + j] += act->aw[li][h][tt] * d_ao[hs + j];
          }
        }
        float dot = 0;
        for (int tt = 0; tt < seq_len; tt++)
          dot += d_aw[tt] * act->aw[li][h][tt];
        float d_al[BLOCK_SIZE];
        for (int tt = 0; tt < seq_len; tt++)
          d_al[tt] = act->aw[li][h][tt] * (d_aw[tt] - dot);
        for (int tt = 0; tt < seq_len; tt++) {
          for (int j = 0; j < HEAD_DIM; j++) {
            d_q[hs + j] += d_al[tt] * kv_keys[li][tt][hs + j] * scale;
            dk_accum[li][tt][hs + j] += d_al[tt] * act->q[li][hs + j] * scale;
          }
        }
      }

      float d_xn[N_EMBD];
      memset(d_xn, 0, sizeof(d_xn));
      linear_bwd_x(attn_wq[li], d_q, N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], d_q, N_EMBD, N_EMBD, d_attn_wq[li]);
      linear_bwd_x(attn_wk[li], dk_accum[li][pos], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], dk_accum[li][pos], N_EMBD, N_EMBD,
                   d_attn_wk[li]);
      linear_bwd_x(attn_wv[li], dv_accum[li][pos], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], dv_accum[li][pos], N_EMBD, N_EMBD,
                   d_attn_wv[li]);

      float d_x_in[N_EMBD];
      memset(d_x_in, 0, sizeof(d_x_in));
      rmsnorm_bwd(act->x_in[li], act->rms_scale_attn[li], d_xn, N_EMBD, d_x_in);
      for (int i = 0; i < N_EMBD; i++)
        dx[i] = dx[i] + d_x_in[i];
    }

    float d_embed[N_EMBD];
    memset(d_embed, 0, sizeof(d_embed));
    rmsnorm_bwd(act->x_embed, act->rms_scale_init, dx, N_EMBD, d_embed);

    int tok = tokens[pos];
    for (int i = 0; i < N_EMBD; i++) {
      d_wte[tok * N_EMBD + i] += d_embed[i];
      d_wpe[pos * N_EMBD + i] += d_embed[i];
    }
  }
}

/* ------------------------------------------------------------------ */
/*  Gradient clipping (by global norm)                                */
/* ------------------------------------------------------------------ */
static float clip_gradients(float max_norm) {
  float total_norm = 0;
  int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
  int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
  
  for (int i = 0; i < es; i++) {
    total_norm += d_wte[i] * d_wte[i];
    total_norm += d_lm_head[i] * d_lm_head[i];
  }
  for (int i = 0; i < ps; i++)
    total_norm += d_wpe[i] * d_wpe[i];
  
  for (int l = 0; l < N_LAYER; l++) {
    for (int i = 0; i < as; i++) {
      total_norm += d_attn_wq[l][i] * d_attn_wq[l][i];
      total_norm += d_attn_wk[l][i] * d_attn_wk[l][i];
      total_norm += d_attn_wv[l][i] * d_attn_wv[l][i];
      total_norm += d_attn_wo[l][i] * d_attn_wo[l][i];
    }
    for (int i = 0; i < ms; i++) {
      total_norm += d_mlp_fc1[l][i] * d_mlp_fc1[l][i];
      total_norm += d_mlp_fc2[l][i] * d_mlp_fc2[l][i];
    }
  }
  
  total_norm = sqrtf(total_norm);
  
  if (total_norm > max_norm) {
    float scale = max_norm / total_norm;
    for (int i = 0; i < es; i++) {
      d_wte[i] *= scale;
      d_lm_head[i] *= scale;
    }
    for (int i = 0; i < ps; i++)
      d_wpe[i] *= scale;
    for (int l = 0; l < N_LAYER; l++) {
      for (int i = 0; i < as; i++) {
        d_attn_wq[l][i] *= scale;
        d_attn_wk[l][i] *= scale;
        d_attn_wv[l][i] *= scale;
        d_attn_wo[l][i] *= scale;
      }
      for (int i = 0; i < ms; i++) {
        d_mlp_fc1[l][i] *= scale;
        d_mlp_fc2[l][i] *= scale;
      }
    }
  }
  
  return total_norm;
}

/* ------------------------------------------------------------------ */
/*  Adam update helper                                                */
/* ------------------------------------------------------------------ */
static void adam_update(float *p, float *g, float *m, float *v, int sz,
                        float lr, float b1, float b2, float eps, int step) {
  float b1c = 1.0f - powf(b1, step + 1);
  float b2c = 1.0f - powf(b2, step + 1);
  for (int i = 0; i < sz; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / b1c) / (sqrtf(v[i] / b2c) + eps);
    g[i] = 0;
  }
}

/* ------------------------------------------------------------------ */
/*  Advanced sampling: top-k and top-p (nucleus) sampling             */
/* ------------------------------------------------------------------ */
typedef struct {
  int idx;
  float prob;
} IndexProb;

static int cmp_prob_desc(const void *a, const void *b) {
  float diff = ((IndexProb *)b)->prob - ((IndexProb *)a)->prob;
  return (diff > 0) ? 1 : (diff < 0 ? -1 : 0);
}

static int sample_topk(const float *probs, int n, int k) {
  IndexProb *pairs = (IndexProb *)malloc(n * sizeof(IndexProb));
  for (int i = 0; i < n; i++) {
    pairs[i].idx = i;
    pairs[i].prob = probs[i];
  }
  qsort(pairs, n, sizeof(IndexProb), cmp_prob_desc);
  
  float sum = 0;
  int actual_k = k < n ? k : n;
  for (int i = 0; i < actual_k; i++)
    sum += pairs[i].prob;
  
  float r = (float)rng_uniform() * sum, cum = 0;
  int result = pairs[actual_k - 1].idx;
  for (int i = 0; i < actual_k; i++) {
    cum += pairs[i].prob;
    if (r < cum) {
      result = pairs[i].idx;
      break;
    }
  }
  
  free(pairs);
  return result;
}

static int sample_topp(const float *probs, int n, float p) {
  IndexProb *pairs = (IndexProb *)malloc(n * sizeof(IndexProb));
  for (int i = 0; i < n; i++) {
    pairs[i].idx = i;
    pairs[i].prob = probs[i];
  }
  qsort(pairs, n, sizeof(IndexProb), cmp_prob_desc);
  
  float cum = 0;
  int cutoff = n;
  for (int i = 0; i < n; i++) {
    cum += pairs[i].prob;
    if (cum >= p) {
      cutoff = i + 1;
      break;
    }
  }
  
  float r = (float)rng_uniform() * cum, sum = 0;
  int result = pairs[cutoff - 1].idx;
  for (int i = 0; i < cutoff; i++) {
    sum += pairs[i].prob;
    if (r < sum) {
      result = pairs[i].idx;
      break;
    }
  }
  
  free(pairs);
  return result;
}

static int weighted_choice(const float *w, int n) {
  float total = 0;
  for (int i = 0; i < n; i++)
    total += w[i];
  float r = (float)rng_uniform() * total, cum = 0;
  for (int i = 0; i < n; i++) {
    cum += w[i];
    if (r < cum)
      return i;
  }
  return n - 1;
}

/* ------------------------------------------------------------------ */
/*  Validation loss calculation                                       */
/* ------------------------------------------------------------------ */
static float compute_val_loss(int num_val_samples) {
  if (num_val_docs == 0)
    return 0;
  
  float total_loss = 0;
  int total_tokens = 0;
  int samples = num_val_samples < num_val_docs ? num_val_samples : num_val_docs;
  
  for (int si = 0; si < samples; si++) {
    char *doc = docs[num_train_docs + si];
    int doc_len = (int)strlen(doc);
    
    int tokens[MAX_DOC_LEN + 2];
    tokens[0] = BOS;
    for (int i = 0; i < doc_len; i++)
      tokens[i + 1] = char_to_id(doc[i]);
    tokens[doc_len + 1] = BOS;
    int n = BLOCK_SIZE < (doc_len + 1) ? BLOCK_SIZE : (doc_len + 1);
    
    PosActs tmp_act;
    for (int pos = 0; pos < n; pos++) {
      float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
      gpt_forward(tokens[pos], pos, logits, &tmp_act);
      softmax_fwd(logits, vocab_size, probs);
      total_loss += -logf(probs[tokens[pos + 1]] + 1e-30f);
      total_tokens++;
    }
    
    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));
  }
  
  return total_loss / total_tokens;
}

/* ------------------------------------------------------------------ */
/*  Main: training + inference                                        */
/* ------------------------------------------------------------------ */
int main(void) {
  printf("=== Enhanced GPT Training ===\n\n");
  
  /* Configuration */
  float val_split = 0.1f;
  float lr = 1e-2f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
  float grad_clip = 1.0f;
  int warmup_steps = 100;
  int num_steps = 5000;
  int checkpoint_interval = 500;
  int val_interval = 100;
  int num_val_samples = 50;
  
  /* Load and prepare dataset */
  load_dataset("input.txt", val_split);
  
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

  printf("Total docs: %d (train: %d, val: %d)\n", num_docs, num_train_docs, num_val_docs);
  build_tokenizer();
  printf("Vocab size: %d\n", vocab_size);
  init_params();
  
  /* Try to load checkpoint */
  int start_step = load_checkpoint("checkpoint.bin");
  
  /* Training statistics */
  float best_val_loss = 1e9f;
  float running_loss = 0;
  int running_count = 0;
  clock_t start_time = clock();

  printf("\nStarting training from step %d...\n", start_step);
  
  for (int step = start_step; step < num_steps; step++) {
    char *doc = docs[step % num_train_docs];
    int doc_len = (int)strlen(doc);

    int tokens[MAX_DOC_LEN + 2], targets[BLOCK_SIZE];
    tokens[0] = BOS;
    for (int i = 0; i < doc_len; i++)
      tokens[i + 1] = char_to_id(doc[i]);
    tokens[doc_len + 1] = BOS;
    int n = BLOCK_SIZE < (doc_len + 1) ? BLOCK_SIZE : (doc_len + 1);

    float total_loss = 0;
    float logits[MAX_CHARS + 1];
    for (int pos = 0; pos < n; pos++) {
      targets[pos] = tokens[pos + 1];
      gpt_forward(tokens[pos], pos, logits, &saved[pos]);
      softmax_fwd(logits, vocab_size, saved_probs[pos]);
      total_loss += -logf(saved_probs[pos][targets[pos]] + 1e-30f);
    }
    float loss = total_loss / n;

    gpt_backward(n, tokens, targets);
    
    /* Gradient clipping */
    float grad_norm = clip_gradients(grad_clip);

    /* Learning rate with warmup + cosine decay */
    float lr_mult = 1.0f;
    if (step < warmup_steps) {
      lr_mult = (float)(step + 1) / (float)warmup_steps;
    }
    float lr_t = lr * lr_mult * 0.5f * 
                 (1.0f + cosf((float)M_PI * step / (float)num_steps));
    
    /* Adam update */
    int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
    int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
    adam_update(wte, d_wte, adam_m_wte, adam_v_wte, es, lr_t, b1, b2, eps, step);
    adam_update(wpe, d_wpe, adam_m_wpe, adam_v_wpe, ps, lr_t, b1, b2, eps, step);
    adam_update(lm_head, d_lm_head, adam_m_lm, adam_v_lm, es, lr_t, b1, b2, eps, step);
    for (int i = 0; i < N_LAYER; i++) {
      adam_update(attn_wq[i], d_attn_wq[i], adam_m_wq[i], adam_v_wq[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(attn_wk[i], d_attn_wk[i], adam_m_wk[i], adam_v_wk[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(attn_wv[i], d_attn_wv[i], adam_m_wv[i], adam_v_wv[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(attn_wo[i], d_attn_wo[i], adam_m_wo[i], adam_v_wo[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(mlp_fc1[i], d_mlp_fc1[i], adam_m_fc1[i], adam_v_fc1[i], ms,
                  lr_t, b1, b2, eps, step);
      adam_update(mlp_fc2[i], d_mlp_fc2[i], adam_m_fc2[i], adam_v_fc2[i], ms,
                  lr_t, b1, b2, eps, step);
    }

    /* Track running loss */
    running_loss += loss;
    running_count++;
    
    /* Progress reporting */
    if ((step + 1) % 50 == 0) {
      float avg_loss = running_loss / running_count;
      clock_t current = clock();
      float elapsed = (float)(current - start_time) / CLOCKS_PER_SEC;
      float tokens_per_sec = (running_count * BLOCK_SIZE) / elapsed;
      float eta = (num_steps - step - 1) * elapsed / (step - start_step + 1);
      
      printf("step %4d/%4d | loss %.4f (avg %.4f) | lr %.6f | grad %.3f | "
             "%.0f tok/s | eta %.0fs\n",
             step + 1, num_steps, loss, avg_loss, lr_t, grad_norm,
             tokens_per_sec, eta);
      
      running_loss = 0;
      running_count = 0;
      start_time = current;
    }
    
    /* Validation */
    if ((step + 1) % val_interval == 0 && num_val_docs > 0) {
      float val_loss = compute_val_loss(num_val_samples);
      printf("  [VAL] loss %.4f", val_loss);
      if (val_loss < best_val_loss) {
        best_val_loss = val_loss;
        printf(" (best!)");
        save_checkpoint("best_model.bin", step + 1, best_val_loss);
      }
      printf("\n");
    }
    
    /* Checkpointing */
    if ((step + 1) % checkpoint_interval == 0) {
      save_checkpoint("checkpoint.bin", step + 1, loss);
    }
  }

  printf("\nTraining complete! Best val loss: %.4f\n", best_val_loss);

  /* ---- Inference with different sampling strategies ---- */
  printf("\n=== Inference ===\n");
  
  /* Standard sampling */
  printf("\n--- Standard sampling (temp=0.8) ---\n");
  for (int si = 0; si < 5; si++) {
    char sample[BLOCK_SIZE + 1];
    int slen = 0, token_id = BOS;
    PosActs tmp_act;
    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
      float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
      gpt_forward(token_id, pos, logits, &tmp_act);
      for (int i = 0; i < vocab_size; i++)
        logits[i] *= 1.25f;  /* temp = 0.8 -> inv_t = 1.25 */
      softmax_fwd(logits, vocab_size, probs);
      token_id = weighted_choice(probs, vocab_size);
      if (token_id == BOS)
        break;
      if (token_id < num_uchars)
        sample[slen++] = uchars_arr[token_id];
    }
    sample[slen] = '\0';
    printf("%2d: %s\n", si + 1, sample);
    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));
  }
  
  /* Top-k sampling */
  printf("\n--- Top-k sampling (k=5, temp=1.0) ---\n");
  for (int si = 0; si < 5; si++) {
    char sample[BLOCK_SIZE + 1];
    int slen = 0, token_id = BOS;
    PosActs tmp_act;
    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
      float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
      gpt_forward(token_id, pos, logits, &tmp_act);
      softmax_fwd(logits, vocab_size, probs);
      token_id = sample_topk(probs, vocab_size, 5);
      if (token_id == BOS)
        break;
      if (token_id < num_uchars)
        sample[slen++] = uchars_arr[token_id];
    }
    sample[slen] = '\0';
    printf("%2d: %s\n", si + 1, sample);
    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));
  }
  
  /* Top-p (nucleus) sampling */
  printf("\n--- Top-p sampling (p=0.9, temp=1.0) ---\n");
  for (int si = 0; si < 5; si++) {
    char sample[BLOCK_SIZE + 1];
    int slen = 0, token_id = BOS;
    PosActs tmp_act;
    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
      float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
      gpt_forward(token_id, pos, logits, &tmp_act);
      softmax_fwd(logits, vocab_size, probs);
      token_id = sample_topp(probs, vocab_size, 0.9f);
      if (token_id == BOS)
        break;
      if (token_id < num_uchars)
        sample[slen++] = uchars_arr[token_id];
    }
    sample[slen] = '\0';
    printf("%2d: %s\n", si + 1, sample);
    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));
  }

  /* Cleanup */
  free(wte);
  free(d_wte);
  free(adam_m_wte);
  free(adam_v_wte);
  free(wpe);
  free(d_wpe);
  free(adam_m_wpe);
  free(adam_v_wpe);
  free(lm_head);
  free(d_lm_head);
  free(adam_m_lm);
  free(adam_v_lm);
  for (int i = 0; i < N_LAYER; i++) {
    free(attn_wq[i]);
    free(d_attn_wq[i]);
    free(adam_m_wq[i]);
    free(adam_v_wq[i]);
    free(attn_wk[i]);
    free(d_attn_wk[i]);
    free(adam_m_wk[i]);
    free(adam_v_wk[i]);
    free(attn_wv[i]);
    free(d_attn_wv[i]);
    free(adam_m_wv[i]);
    free(adam_v_wv[i]);
    free(attn_wo[i]);
    free(d_attn_wo[i]);
    free(adam_m_wo[i]);
    free(adam_v_wo[i]);
    free(mlp_fc1[i]);
    free(d_mlp_fc1[i]);
    free(adam_m_fc1[i]);
    free(adam_v_fc1[i]);
    free(mlp_fc2[i]);
    free(d_mlp_fc2[i]);
    free(adam_m_fc2[i]);
    free(adam_v_fc2[i]);
  }
  
  printf("\nDone!\n");
  return 0;
}
