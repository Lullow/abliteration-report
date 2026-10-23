# The Dark Side: How Abliteration Removes Refusal from Language Models

A white-hat look at methods that strip the safety behaviour out of open language models: abliteration, "uncensoring" and the tool Heretic. The goal is to understand what we need to protect against. The report explains how abliteration works and why it works so well, and it summarizes what published measurements show about its effect.

## Contents
1. [What abliteration is](#1-what-abliteration-is)
2. [Why it works](#2-why-it-works)
3. [What the measurements show](#3-what-the-measurements-show)
4. [Risks](#4-risks)
5. [Defences](#5-defences)
6. [Reflection](#6-reflection)
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

## 4. Risks

### Abliterated models are easy to get

Ready-made abliterated versions of popular models are published openly. huihui-ai alone offers abliterated versions of the whole Qwen3 family on Ollama, from 0.6B to 235B parameters. With Heretic, making a new one is a single command. The model cards admit the limits: huihui-ai calls its method a "crude, proof-of-concept implementation" and says that its own test result, 100 % on 320 harmful instructions, does not represent actual results. The models are downloaded and used anyway.

### Text alone is limited, agents are not

In a chat window, a model can only write text, and a person has to decide to act on it. The risk grows when the model gets tools: code execution, email, file access, the internet. In an agent, refusal is one of the few things standing between an instruction and an action. That includes instructions the user never wrote, planted in a web page, email or document the agent reads (*prompt injection*). An abliterated model in an agent carries out whatever it is told, by whoever manages to tell it.

### A manipulated model is hard to spot

Section 3 shows that an orthogonalized model answers harmless prompts almost exactly like the original and scores about the same on standard benchmarks. Ordinary testing will not reveal a swap. If a team downloads a model under a familiar name from an untrusted source, or someone replaces the file somewhere in a pipeline, the only visible difference is a missing refusal, and nobody looks for that until it matters.

### Who is responsible?

When an abliterated model causes harm, the responsibility is spread out. The original developer trained the safety in. The person who removed it published the result with a disclaimer; huihui-ai's model card recommends research or controlled environments and disclaims responsibility for how the model is used. The person who deployed it decided what it could reach, and the user decided what to ask. In practice the weight falls on whoever deploys the model, because only they control what it is connected to.

## 5. Defences

The starting point: **assume the model's own refusals can be removed, and place defences where the model cannot reach them.**

### Guardrails outside the model

A separate classifier can check what goes into the model and what comes out of it. Llama Guard, the same classifier Arditi et al. use to judge answers, is one example. NVIDIA's NeMo Guardrails, which wraps an LLM application in programmable rules, is another. Because they are separate from the model, abliterating the model does not remove them. This report has its own example: the safety classifier that stopped its planned experiment (section 6).

The limit is who controls them. Guardrails protect a service from its users and from injected content. They do not help when the person running the model is the threat: whoever runs an open model on their own machine can leave out the classifier as easily as they can download an abliterated model.

### Limit what an agent can do

- **Least privilege.** Give an agent only the tools it needs, no internet access unless the task requires it, and only the data it needs to know.
- **Sandboxing.** Run agents in an isolated environment. This homework's own rules are an example: offline, no function calling. Without tools, even an abliterated model can only write text.
- **Human approval** for actions that cannot be undone.
- **Enforcement outside the agent.** NVIDIA's Open Agent Safety Platform (announced 28 September 2026) takes this furthest. OpenShell is a runtime that sets boundaries for agents, traces what they do and enforces policy outside both the model and the agent harness. Sentry is a watchdog that runs on separate hardware (BlueField-4 DPUs), monitors agent behaviour and can quarantine an agent that tries to leave its boundaries. The limits sit where neither the model nor its harness can switch them off.

### Know where the model comes from

- Download models from the original publisher, and check the file's digest against the one the publisher lists. Ollama and Hugging Face both show content digests.
- Test the exact file that will be deployed with a refusal benchmark such as HarmBench or JailbreakBench. Section 3 shows that general benchmarks will not catch abliteration, but a refusal benchmark will.

### Do not count on the system prompt

A safety instruction in the system prompt brought much of the refusal back for Llama-2, but did nothing for Qwen, and the effect varied widely between prompts (section 3). It is a useful layer, not a guarantee.

No single layer is enough on its own. The defences work together, so that one failing layer does not open the whole system.

## 6. Reflection

### Two sides of removing refusal

Removing refusal is not only a tool for attackers. Arditi et al.'s research, which explains why safety training is so brittle, was only possible by manipulating refusal. Red teams need to know what an unrestricted model does to test their defences. And some people use abliterated models simply because the original refuses too much. The risks in section 4 are real, but so are these uses.

### Alignment can be dangerous too

My teacher's point was that alignment has risks of its own, and the research supports this in two ways.

**Over-refusal.** Arditi et al. show that adding the refusal direction makes a model refuse harmless requests. Over-refusal is the same mechanism firing at the wrong time. A model that refuses a medical question, a history question about a war crime or a security course exercise is not safer, only less useful, and it pushes people toward models with no limits at all.

**Models that act on their own.** In Anthropic's study *Agentic Misalignment* (June 2025), 16 models from several developers worked as email agents in a simulated company. Faced with a threat of being replaced, or a conflict between their goal and the company's, several of them blackmailed an executive or leaked confidential documents. Explicit instructions not to do so reduced the behaviour but did not remove it: for Claude Opus 4, blackmail fell from 96 % to 37 %. The authors have not seen this in real deployments. Still, a model that acts on its own judgement with real tools can do harm while believing it does the right thing. Neither removing nor adding refusal solves that. The question is not only whether a model says no, but what it is allowed to do.

### Why there is no experiment in this report

The plan was a small offline comparison of `qwen3:8b` with `huihui_ai/qwen3-abliterated:8b`, logging only whether each model refused. I used an AI assistant (Claude Code) for the work. When it started writing the test prompts, its answer was stopped by a safety classifier. I chose not to work around it, neither by rephrasing the request nor by doing the same thing by hand from the assistant's instructions. The experiment was dropped and the downloaded model removed.

Was the stop reasonable? My teacher allowed experiments in a sandbox, and the classifier had no way of knowing that. It judges the text in front of it, just like the models in this report: neither the classifier nor a model with refusals can see who is asking or why. A filter that cannot verify the context has to choose between stopping too much and stopping too little, and this one stopped too much from my teacher's point of view. I still think it was reasonable. The cost was one section of my own measurements, which published research covers better anyway.

The episode is also a live example of section 5. The classifier sits outside the model, and as a user I could not abliterate it away. It also shows the limit of such guardrails: the classifier only existed because I used a hosted service. With a model running on my own machine, there would have been nothing to stop me.

## Sources

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W. & Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Labonne, M. (2024). *Uncensor any LLM with abliteration.* [Hugging Face blog](https://huggingface.co/blog/mlabonne/abliteration)
- Weidmann, P. E. *Heretic.* [heretic-project.org](https://heretic-project.org/), [GitHub](https://github.com/p-e-w/heretic)
- Anthropic (2025). *Agentic Misalignment: How LLMs could be insider threats.* [anthropic.com](https://www.anthropic.com/research/agentic-misalignment)
- NVIDIA (2026). *Open Agent Safety Platform.* [Press release](https://nvidianews.nvidia.com/news/open-agent-safety-platform)
- NVIDIA. *NeMo Guardrails.* [GitHub](https://github.com/NVIDIA/NeMo-Guardrails)
- huihui-ai. *Qwen3-8B-abliterated.* [Hugging Face](https://huggingface.co/huihui-ai/Qwen3-8B-abliterated), [Ollama](https://ollama.com/huihui_ai/qwen3-abliterated)
