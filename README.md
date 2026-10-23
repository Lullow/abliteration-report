# The Dark Side: How Abliteration Removes Refusal from Language Models

A white-hat look at methods that strip the safety behaviour out of open language models: abliteration, "uncensoring" and the tool Heretic. The goal is to understand what we need to protect against. The report explains how abliteration works and why it works so well, and it compares `qwen3:8b` with an abliterated version of the same base model in a small offline experiment.

## Contents
1. [What abliteration is](#1-what-abliteration-is)
2. [Why it works](#2-why-it-works)
3. The experiment
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

After this, no part of the model can write anything along $\hat{r}$. Arditi et al. show that this is equivalent to directional ablation at every layer. The result is an ordinary model file that can be shared and run like any other, which is how models such as `huihui_ai/qwen3-abliterated:8b` (used in the experiment below) are distributed.

### Heretic: abliteration on autopilot

[Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann automates the whole process. It computes a direction per layer in the same difference-of-means way, then lets an optimizer (Optuna's TPE) search for the best settings: which direction to use and how strongly to ablate at each layer, tuned separately for the attention and MLP parts. The optimizer minimizes two things at once: the number of refusals on harmful prompts and the KL divergence from the original model on harmless ones. On `gemma-3-12b-it`, the author reports 3/100 refusals for Heretic as well as for two hand-made abliterations, but a KL divergence of 0.16 against 0.45 and 1.04. That is the same loss of refusal with less damage to the rest of the model. These are the author's own measurements.

## 2. Why it works

**Refusal is low-dimensional.** Arditi et al. found a single refusal direction in each of 13 open chat models of up to 72B parameters. The evidence is causal in both directions: removing the direction makes a model answer harmful requests, and adding it makes the model refuse harmless ones. Later layers seem to read one signal, "refuse this", and act on it. Remove the signal and they never see it.

**Safety fine-tuning adds a gate, not a gap in knowledge.** The knowledge needed to answer is still in the model. Refusal works like a gate in front of it, and that gate hangs on one direction. Arditi et al. also find the direction in base models before any chat fine-tuning, which suggests that safety training hooks into a representation that already exists rather than building a new one. In their words, the findings "underscore the brittleness of current safety fine-tuning methods".

**The rest of the model mostly survives.** On MMLU, ARC and GSM8K (standard tests of knowledge, reasoning and arithmetic), the orthogonalized models score about the same as the originals, within the 99 % confidence intervals for all models except Qwen 7B and Yi 34B. TruthfulQA drops consistently. Labonne saw benchmark scores fall on his abliterated model and recovered most of the loss with further fine-tuning (DPO). Heretic's two objectives exist for the same reason: removing refusal is easy, and the hard part is to do it without damaging the rest.

**It needs the weights, and nothing else.** Abliteration is a *white-box* attack: it requires access to the model's weights, a few hundred prompts and a GPU that can run the model. Models behind an API cannot be abliterated this way. Every open-weight model can, which is why any safety argument for an open model cannot rest on its refusals alone.

<!-- Sections 3–6 follow: experiment, risks, defences, reflection. -->

## Sources

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W. & Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Labonne, M. (2024). *Uncensor any LLM with abliteration.* [Hugging Face blog](https://huggingface.co/blog/mlabonne/abliteration)
- Weidmann, P. E. *Heretic.* [heretic-project.org](https://heretic-project.org/), [GitHub](https://github.com/p-e-w/heretic)
- huihui-ai. *Qwen3-8B-abliterated.* [Hugging Face](https://huggingface.co/huihui-ai/Qwen3-8B-abliterated), [Ollama](https://ollama.com/huihui_ai/qwen3-abliterated)
