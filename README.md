# The Dark Side: How Abliteration Removes Refusal from Language Models

A white-hat look at methods that strip the safety behaviour out of open language models: abliteration, "uncensoring" and the tool Heretic. The goal is to understand what we need to protect against. The report explains how abliteration works and why it works so well, and it summarizes what published measurements show about its effect.

## Contents
1. [What abliteration is](#1-what-abliteration-is)
2. [Why it works](#2-why-it-works)
3. [What the measurements show](#3-what-the-measurements-show)
4. Risks
5. Defences
6. Reflection
7. [Sources](#sources)

## 1. What abliteration is

Chat models are fine-tuned to answer ordinary requests and to refuse harmful ones. Abliteration removes the refusal without any retraining. It edits the weights of the model directly, guided by one vector that the model uses to represent "this should be refused". The method comes from Arditi et al. (2024), and Maxime Labonne's blog post is the practical guide most implementations follow.

The idea in one sentence: **first find a pattern inside the model that goes with refusal, then test whether that pattern actually drives the behaviour, and finally remove it for good.**

### Background: the residual stream

A transformer passes one vector per token from layer to layer. This vector is the *residual stream*. Each layer reads from it and adds its own output back to it, like workers along a conveyor belt. In Qwen3-8B the vector has 4096 dimensions. A *direction* is simply a vector in this 4096-dimensional space.

### Step 1: find candidate directions

The model is run on two sets of instructions, one harmful and one harmless (Arditi et al. use 128 of each), and the residual stream is recorded at the tokens that follow the instruction in the chat template. For each layer and each of those token positions, the mean activation over the harmless set is subtracted from the mean over the harmful set:

$$r = \mu_{\text{harmful}} - \mu_{\text{harmless}}$$

Each difference is a *candidate* refusal direction. At this point it only correlates with refusal. Nothing yet shows that it causes it.

### Step 2: choose the direction that drives refusal

The candidates are tested on a separate validation set (32 instructions of each kind), so that the choice is not fitted to the same prompts the directions came from. Arditi et al. keep the direction that best meets three criteria:

1. **Removing it stops refusal.** Ablated, the model refuses the harmful validation prompts least.
2. **Adding it causes refusal.** Added, it makes the model refuse harmless prompts.
3. **Removing it changes little else.** On harmless prompts, the model's output distribution barely moves (KL divergence, a measure of how far two probability distributions differ, below 0.1).

They also skip the last 20 % of layers. A direction found that close to the output would mostly block the tokens a refusal starts with rather than the concept behind it. Labonne's guide does the same selection by hand, reading what each candidate produces.

### Step 3: two interventions

- **Activation addition** adds $r$ at a single layer, at every token position. One layer is enough, because the residual stream carries the addition on to every later layer.
- **Directional ablation** removes the component along the unit vector $\hat{r}$ from every activation $x$, at every layer and every token position:

$$x' = x - \hat{r}\hat{r}^{\top}x$$

Ablation has to happen everywhere, because any later layer could write the direction back into the stream if it were only removed once.

### Step 4: make it permanent with weight orthogonalization

Instead of hooking into the model every time it runs, the direction can be removed from the weights. Every matrix that writes to the residual stream (the token and positional embeddings, the attention output matrices and the MLP output matrices, plus their biases) is orthogonalized against $\hat{r}$:

$$W' = W - \hat{r}\hat{r}^{\top}W$$

After this, no part of the model can write anything along $\hat{r}$. Arditi et al. show that this is equivalent to directional ablation at every layer. The result is an ordinary model file that can be shared and run like any other, which is how models such as `huihui_ai/qwen3-abliterated:8b` are distributed.

### Heretic: abliteration on autopilot

[Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann automates the whole process. It computes a direction per layer in the same difference-of-means way, then lets an optimizer (Optuna's TPE) search for the best settings: which direction to use and how strongly to ablate at each layer, tuned separately for the attention and MLP parts. The optimizer minimizes two things at once: the number of refusals on harmful prompts and the KL divergence from the original model on harmless ones. On `gemma-3-12b-it`, the author reports 3/100 refusals for Heretic as well as for two hand-made abliterations, but a KL divergence of 0.16 against 0.45 and 1.04. That is the same loss of refusal with less damage to the rest of the model. These are the author's own measurements.

## 2. Why it works

**Refusal is low-dimensional.** Arditi et al. found a single refusal direction in each of 13 open chat models of up to 72B parameters. The evidence is causal in both directions: removing the direction makes a model answer harmful requests, and adding it makes the model refuse harmless ones. Later layers seem to read one signal, "refuse this", and act on it. Remove the signal and they never see it.

**Safety fine-tuning adds a gate, not a gap in knowledge.** The knowledge needed to answer is still in the model. Refusal works like a gate in front of it, and that gate hangs on one direction. Arditi et al. also find the direction in base models before any chat fine-tuning, which suggests that safety training hooks into a representation that already exists rather than building a new one. In their words, the findings "underscore the brittleness of current safety fine-tuning methods".

**The rest of the model mostly survives.** Because the edit touches only one direction, standard tests of knowledge, reasoning and arithmetic barely move (numbers in [section 3](#what-the-model-loses)). Removing refusal is easy; the hard part is doing it without damaging the rest, which is why Labonne repaired his model with further fine-tuning (DPO) and why Heretic optimizes for both.

**It needs the weights, and nothing else.** Abliteration is a *white-box* attack: it requires access to the model's weights, a few hundred prompts and a GPU that can run the model. Models behind an API cannot be abliterated this way. Every open-weight model can, which is why any safety argument for an open model cannot rest on its refusals alone.

## 3. What the measurements show

This section collects published measurements. An experiment of our own was planned but not carried out; section 6 explains why. All numbers below come from Arditi et al. (2024) unless stated otherwise.

### How refusal is measured

Arditi et al. score every answer in two ways:

- **Refusal score:** 1 if the answer contains a typical refusal phrase such as "I'm sorry" or "As an AI", otherwise 0.
- **Safety score:** whether a separate classifier model, Llama Guard 2, labels the answer as safe or unsafe.

Neither is enough on its own. An answer can avoid every refusal phrase and still contain nothing harmful, and an answer can open with an apology and then comply anyway. The test sets are 100 harmful instructions from JailbreakBench (10 categories), 100 harmless instructions from Alpaca, and the 159 standard behaviours of HarmBench. HarmBench reports an *attack success rate* (ASR): the share of prompts for which its own classifier judges the answer harmful.

### Removing and adding the direction

- **Ablation removes refusal.** With the direction ablated, refusal falls and unsafe answers rise in all 13 models.
- **Addition creates refusal.** With the direction added, the models start refusing harmless requests.
- **The edit removes the decision, not the words.** Asked directly, the orthogonalized model can still write refusal phrases, so the edit is not just a filter on certain tokens.
- **Harmless behaviour is unchanged.** On harmless instructions the orthogonalized model answers almost exactly like the original, often word for word.

### Compared with other jailbreaks

On HarmBench, weight orthogonalization ("Ortho") is compared with a direct request without any attack, and with or without the model's default system prompt:

| Model | Direct request, no attack | Ortho, default system prompt | Ortho, no system prompt |
|---|---|---|---|
| Llama-2 7B | 0.0 % | 22.6 % | 79.9 % |
| Llama-2 13B | 0.5 % | 6.9 % | 61.0 % |
| Llama-2 70B | 0.0 % | 4.4 % | 62.9 % |

For the Qwen chat models (7B, 14B and 72B), Ortho reaches an ASR of about 75–84 % with or without the system prompt, against 7–10 % for a direct request. For Qwen it is on par with GCG, an attack that optimizes a separate adversarial suffix for every single prompt.

For Llama-2 the system prompt matters a great deal. Its default system prompt explicitly tells the model to avoid harmful content, and with it in place much of the refusal comes back. The effect depends on the model, though. Qwen's default system prompt has no safety instructions, but giving Qwen the Llama-2 prompt made no significant difference either. Across 12 different system prompts, Llama-2 7B's ASR varied widely (mean 30.0 %, standard deviation 23.3 %), while Qwen 7B stayed high and stable (mean 76.7 %, standard deviation 5.9 %). Arditi et al. suggest that Llama-2 simply responds more strongly to system-level instructions, and leave a closer study to future work.

### What the model loses

Standard benchmarks barely move. A selection (original → orthogonalized, accuracy in %):

| Model | MMLU | ARC | GSM8K | TruthfulQA |
|---|---|---|---|---|
| Llama-3 8B | 65.8 → 65.0 | 62.4 → 62.3 | 75.9 → 74.3 | 51.7 → 48.3 |
| Llama-3 70B | 79.9 → 79.8 | 71.8 → 71.5 | 91.2 → 90.8 | 61.8 → 59.5 |
| Qwen 7B | 56.8 → 54.8 | 51.7 → 52.0 | 48.1 → 41.8 | 51.6 → 47.9 |
| Qwen 72B | 77.2 → 76.5 | 67.6 → 67.2 | 75.5 → 76.3 | 56.4 → 55.0 |

Qwen 7B is the clearest exception, with significant drops on MMLU and GSM8K. TruthfulQA falls for every model, by 1–6 points. Arditi et al. note that the dataset has categories close to refusal territory, and that earlier work saw the same drop when safety fine-tuning was removed.

Heretic's author measures the damage differently, as KL divergence from the original model on harmless prompts (`gemma-3-12b-it`):

| Model | Refusals on harmful prompts | KL divergence |
|---|---|---|
| Original | 97/100 | 0 |
| mlabonne abliteration | 3/100 | 1.04 |
| huihui-ai abliteration | 3/100 | 0.45 |
| Heretic | 3/100 | 0.16 |

All three remove refusal equally well. How much else changes depends on how carefully the ablation is tuned.

### In short

Removing refusal costs little in capability, works across model families and sizes, and for Qwen it is as effective as GCG, which has to be optimized separately for every prompt. The only thing in these measurements that brought refusal back was not in the weights: a safety instruction in the system prompt, and only for some models. A defence that lives in the model's own behaviour can be edited away. That points to where defences have to be placed (section 5).

<!-- Sections 4–6 follow: risks, defences, reflection. -->

## Sources

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W. & Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Labonne, M. (2024). *Uncensor any LLM with abliteration.* [Hugging Face blog](https://huggingface.co/blog/mlabonne/abliteration)
- Weidmann, P. E. *Heretic.* [heretic-project.org](https://heretic-project.org/), [GitHub](https://github.com/p-e-w/heretic)
- huihui-ai. *Qwen3-8B-abliterated.* [Hugging Face](https://huggingface.co/huihui-ai/Qwen3-8B-abliterated), [Ollama](https://ollama.com/huihui_ai/qwen3-abliterated)
