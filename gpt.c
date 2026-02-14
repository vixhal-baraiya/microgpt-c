/*
 * The most atomic way to train and inference a GPT in pure, dependency-free C.
 * Optimized: manual forward/backward, float precision, cache-friendly.
 *
 * Compile: gcc -O3 -march=native -ffast-math -o gpt gpt.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
/*  Dataset loading                                                   */
/* ------------------------------------------------------------------ */
#define MAX_DOCS 85000
#define MAX_DOC_LEN 512
#define MAX_CHARS 128

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int num_docs = 0;

static void load_dataset(const char *filename) {
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
    const float *wr = w + r * nin;
    for (int c = 0; c < nin; c++)
      s += wr[c] * x[c];
    out[r] = s;
  }
}

static inline float rmsnorm_fwd(const float *x, int n, float *out) {
  float ms = 0;
  for (int i = 0; i < n; i++)
    ms += x[i] * x[i];
  ms /= n;
  float scale = 1.0f / sqrtf(ms + 1e-5f);
  for (int i = 0; i < n; i++)
    out[i] = x[i] * scale;
  return scale;
}

static inline void softmax_fwd(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++)
    if (logits[i] > mx)
      mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) {
    probs[i] = expf(logits[i] - mx);
    sum += probs[i];
  }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++)
    probs[i] *= inv;
}

/* ------------------------------------------------------------------ */
/*  Backward building blocks                                          */
/* ------------------------------------------------------------------ */
static inline void linear_bwd_x(const float *restrict w,
                                const float *restrict dout, int nout, int nin,
                                float *restrict dx) {
  for (int c = 0; c < nin; c++) {
    float s = 0;
    for (int r = 0; r < nout; r++)
      s += dout[r] * w[r * nin + c];
    dx[c] += s;
  }
}

static inline void linear_bwd_w(const float *restrict x,
                                const float *restrict dout, int nout, int nin,
                                float *restrict dw) {
  for (int r = 0; r < nout; r++) {
    float dr = dout[r];
    float *dwr = dw + r * nin;
    for (int c = 0; c < nin; c++)
      dwr[c] += dr * x[c];
  }
}

static inline void rmsnorm_bwd(const float *x, float scale, const float *dout,
                               int n, float *dx) {
  float dot = 0;
  for (int i = 0; i < n; i++)
    dot += dout[i] * x[i];
  float coeff = scale * scale * scale / n;
  for (int i = 0; i < n; i++)
    dx[i] += scale * dout[i] - coeff * x[i] * dot;
}

/* ------------------------------------------------------------------ */
/*  GPT forward pass (one token, fills saved acts)                    */
/* ------------------------------------------------------------------ */
static void gpt_forward(int token_id, int pos_id, float *logits_out,
                        PosActs *act) {
  float x[N_EMBD], tmp[MLP_DIM > N_EMBD ? MLP_DIM : N_EMBD];

  for (int i = 0; i < N_EMBD; i++)
    x[i] = wte[token_id * N_EMBD + i] + wpe[pos_id * N_EMBD + i];
  memcpy(act->x_embed, x, sizeof(x));

  act->rms_scale_init = rmsnorm_fwd(x, N_EMBD, x);

  for (int li = 0; li < N_LAYER; li++) {
    memcpy(act->x_in[li], x, sizeof(x));

    float xn[N_EMBD];
    act->rms_scale_attn[li] = rmsnorm_fwd(x, N_EMBD, xn);
    memcpy(act->xn_attn[li], xn, sizeof(xn));

    float q[N_EMBD], k[N_EMBD], v[N_EMBD];
    linear_fwd(xn, attn_wq[li], N_EMBD, N_EMBD, q);
    linear_fwd(xn, attn_wk[li], N_EMBD, N_EMBD, k);
    linear_fwd(xn, attn_wv[li], N_EMBD, N_EMBD, v);
    memcpy(act->q[li], q, sizeof(q));

    memcpy(kv_keys[li][pos_id], k, sizeof(k));
    memcpy(kv_vals[li][pos_id], v, sizeof(v));
    int seq_len = pos_id + 1;
    float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);

    float ao[N_EMBD];
    for (int h = 0; h < N_HEAD; h++) {
      int hs = h * HEAD_DIM;
      float al[BLOCK_SIZE];
      for (int tt = 0; tt < seq_len; tt++) {
        float dot = 0;
        for (int j = 0; j < HEAD_DIM; j++)
          dot += q[hs + j] * kv_keys[li][tt][hs + j];
        al[tt] = dot * scale;
      }
      float mx = al[0];
      for (int tt = 1; tt < seq_len; tt++)
        if (al[tt] > mx)
          mx = al[tt];
      float sm = 0;
      for (int tt = 0; tt < seq_len; tt++) {
        al[tt] = expf(al[tt] - mx);
        sm += al[tt];
      }
      float inv = 1.0f / sm;
      for (int tt = 0; tt < seq_len; tt++)
        al[tt] *= inv;
      for (int tt = 0; tt < seq_len; tt++)
        act->aw[li][h][tt] = al[tt];
      for (int j = 0; j < HEAD_DIM; j++) {
        float s = 0;
        for (int tt = 0; tt < seq_len; tt++)
          s += al[tt] * kv_vals[li][tt][hs + j];
        ao[hs + j] = s;
      }
    }
    memcpy(act->attn_out[li], ao, sizeof(ao));

    linear_fwd(ao, attn_wo[li], N_EMBD, N_EMBD, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + act->x_in[li][i];
    memcpy(act->x_mid[li], x, sizeof(x));

    float xn_m[N_EMBD];
    act->rms_scale_mlp[li] = rmsnorm_fwd(x, N_EMBD, xn_m);
    memcpy(act->xn_mlp[li], xn_m, sizeof(xn_m));

    float h1[MLP_DIM];
    linear_fwd(xn_m, mlp_fc1[li], MLP_DIM, N_EMBD, h1);
    memcpy(act->mlp_pre[li], h1, MLP_DIM * sizeof(float));

    float h2[MLP_DIM];
    for (int i = 0; i < MLP_DIM; i++)
      h2[i] = h1[i] > 0 ? h1[i] * h1[i] : 0;
    memcpy(act->mlp_post[li], h2, MLP_DIM * sizeof(float));

    linear_fwd(h2, mlp_fc2[li], N_EMBD, MLP_DIM, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + act->x_mid[li][i];
  }

  memcpy(act->x_out, x, sizeof(x));
  linear_fwd(x, lm_head, vocab_size, N_EMBD, logits_out);
}

/* ------------------------------------------------------------------ */
/*  Backward pass for all positions                                   */
/* ------------------------------------------------------------------ */
static void gpt_backward(int n, const int *tokens, const int *targets) {
  memset(dk_accum, 0, sizeof(dk_accum));
  memset(dv_accum, 0, sizeof(dv_accum));
  float inv_n = 1.0f / n;

  for (int pos = n - 1; pos >= 0; pos--) {
    PosActs *act = &saved[pos];
    int seq_len = pos + 1;

    float dl[MAX_CHARS + 1];
    for (int i = 0; i < vocab_size; i++)
      dl[i] = (saved_probs[pos][i] - (i == targets[pos] ? 1.0f : 0.0f)) * inv_n;

    float dx[N_EMBD];
    memset(dx, 0, sizeof(dx));
    linear_bwd_x(lm_head, dl, vocab_size, N_EMBD, dx);
    linear_bwd_w(act->x_out, dl, vocab_size, N_EMBD, d_lm_head);

    for (int li = N_LAYER - 1; li >= 0; li--) {
      /* MLP backward */
      float d_h2[MLP_DIM];
      memset(d_h2, 0, sizeof(d_h2));
      linear_bwd_x(mlp_fc2[li], dx, N_EMBD, MLP_DIM, d_h2);
      linear_bwd_w(act->mlp_post[li], dx, N_EMBD, MLP_DIM, d_mlp_fc2[li]);

      float d_h1[MLP_DIM];
      for (int i = 0; i < MLP_DIM; i++)
        d_h1[i] =
            act->mlp_pre[li][i] > 0 ? 2.0f * act->mlp_pre[li][i] * d_h2[i] : 0;

      float d_xn_mlp[N_EMBD];
      memset(d_xn_mlp, 0, sizeof(d_xn_mlp));
      linear_bwd_x(mlp_fc1[li], d_h1, MLP_DIM, N_EMBD, d_xn_mlp);
      linear_bwd_w(act->xn_mlp[li], d_h1, MLP_DIM, N_EMBD, d_mlp_fc1[li]);

      float d_x_mid[N_EMBD];
      memset(d_x_mid, 0, sizeof(d_x_mid));
      rmsnorm_bwd(act->x_mid[li], act->rms_scale_mlp[li], d_xn_mlp, N_EMBD,
                  d_x_mid);
      for (int i = 0; i < N_EMBD; i++)
        dx[i] += d_x_mid[i];

      /* Attention backward */
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
/*  Weighted random choice                                            */
/* ------------------------------------------------------------------ */
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
/*  Main: training + inference                                        */
/* ------------------------------------------------------------------ */
int main(void) {
  load_dataset("input.txt");

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
  build_tokenizer();
  printf("vocab size: %d\n", vocab_size);
  init_params();

  float lr = 1e-2f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
  int num_steps = 5000;

  for (int step = 0; step < num_steps; step++) {
    char *doc = docs[step % num_docs];
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

    float lr_t =
        lr * 0.5f * (1.0f + cosf((float)M_PI * step / (float)num_steps));
    int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
    int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
    adam_update(wte, d_wte, adam_m_wte, adam_v_wte, es, lr_t, b1, b2, eps,
                step);
    adam_update(wpe, d_wpe, adam_m_wpe, adam_v_wpe, ps, lr_t, b1, b2, eps,
                step);
    adam_update(lm_head, d_lm_head, adam_m_lm, adam_v_lm, es, lr_t, b1, b2, eps,
                step);
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

    printf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss);
  }

  /* ---- Inference ---- */
  float temperature = 0.5f;
  printf("\n--- inference ---\n");
  for (int si = 0; si < 20; si++) {
    char sample[BLOCK_SIZE + 1];
    int slen = 0, token_id = BOS;
    PosActs tmp_act;
    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
      float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
      gpt_forward(token_id, pos, logits, &tmp_act);
      float inv_t = 1.0f / temperature;
      for (int i = 0; i < vocab_size; i++)
        logits[i] *= inv_t;
      softmax_fwd(logits, vocab_size, probs);
      token_id = weighted_choice(probs, vocab_size);
      if (token_id == BOS)
        break;
      if (token_id < num_uchars)
        sample[slen++] = uchars_arr[token_id];
    }
    sample[slen] = '\0';
    printf("sample %2d: %s\n", si + 1, sample);
    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));
  }

  /* cleanup */
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
  return 0;
}
