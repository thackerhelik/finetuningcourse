# TinyLlama Pharmacology Assistant: 3-Stage Post-Training Pipeline

An end-to-end post-training pipeline adapting a base `TinyLlama-1.1B` model into a domain-specific, instruction-following pharmacology assistant aligned with preference optimization and deployed locally via Ollama.

> **Attribution & Acknowledgments:**  
> This project was developed as an end-to-end implementation inspired by the **freeCodeCamp** course **[LLM Fine-Tuning Course](https://www.freecodecamp.org/)** instructed by **Sunny Savita**. The foundational progression (Domain Adaptation → SFT → DPO) is modernized here to adhere to current Hugging Face `trl` standards, strict FP16/BF16 mixed-precision stability, and native GGUF/Ollama edge deployment.

---

## Architecture & Post-Training Workflow

```
[Raw Base Model] TinyLlama-1.1B-intermediate-step-1431k-3T
       │
       ▼
[Stage 1: Domain Adaptation] Continued Pre-training on Metformin literature
       │
       ▼
[Stage 2: Instruction Fine-Tuning] QLoRA SFT on conversational QA pairs
       │
       ▼
[Stage 3: Preference Alignment] Direct Preference Optimization (DPO) to correct polarity & hallucinations
       │
       ▼
[Stage 4: Quantization & Deployment] 4-bit GGUF export with custom Ollama Modelfile
```

### Pipeline Overview

| Stage | Objective | Data / Method | Key Artifacts |
| :--- | :--- | :--- | :--- |
| **Stage 1: Domain Adaptation** | Impart foundational domain vocabulary and pharmacology knowledge | Continuous text chunking from scientific literature (`Metformin.pdf`) via Causal Language Modeling | `stage1_domain/merged_model` |
| **Stage 2: Instruction Tuning (SFT)** | Align model behavior to follow structured clinical QA prompts | 4-bit QLoRA with `SFTTrainer` on prompt-completion pairs | `stage2_instruct/merged_model` |
| **Stage 3: Preference Alignment (DPO)** | Suppress pharmacological hallucinations and polarity flips (e.g., AMPK activation vs. inhibition) | Contrastive loss with `DPOTrainer` ($\beta=0.1$) on `(prompt, chosen, rejected)` triplets | `stage3_dpo/merged_model` |
| **Stage 4: Export & Serving** | Enable low-latency, CPU/edge-friendly local inference | `llama.cpp` conversion to 4-bit GGUF (`q4_k_m`) served via Ollama | `metformin-dpo` (Ollama) |

---

## Technical Highlights & Engineering Fixes

* **Zero-Vocab-Drift Tokenization:** Bypasses CUDA `vectorized_gather_kernel` index-out-of-bounds assertions caused by resizing quantized 4-bit embeddings. Uses the model's native `tokenizer.eos_token` (`</s>`) rather than appending ungrounded custom tokens.
* **Numerical Stability in Mixed-Precision:** Enforces standard `float32` casting on trainable LoRA adapters during QLoRA training to prevent PyTorch `GradScaler` AMP unscale exceptions (`_amp_foreach_non_finite_check_and_unscale_cuda`).
* **VRAM-Optimized DPO Alignment:** Configures `ref_model=None` in modern `DPOTrainer`, allowing PEFT to automatically disable the active adapter during evaluation to act as the implicit reference policy ($\pi_{\text{ref}}$), avoiding double-VRAM overhead.
* **Stop-Token Engineering for Small LLMs:** Addresses run-on generation loops common to 1.1B models by injecting `PARAMETER stop "### Instruction:"` directly into the Ollama `Modelfile`.

---

## Repository Structure

```text
├── data/
│   ├── Metformin.pdf                  # Domain-specific medical corpus
│   ├── pharma_instruction_data.csv    # Supervised QA dataset
│   └── pharma_preference_data.csv     # DPO preference triplets (chosen vs. rejected)
│
├── notebooks/
│   ├── 01_domain_adaptation.ipynb     # Stage 1: Continued Pre-Training on domain text
│   ├── 02_instruction_tuning.ipynb    # Stage 2: 4-bit QLoRA SFT on conversational QA
│   └── 03_dpo_alignment.ipynb         # Stage 3: Direct Preference Optimization (DPO)
│
├── export_to_ollama.py                # Automated GGUF conversion, quantization, and Ollama setup
├── .gitignore                         # Strict exclusion of multi-gigabyte safetensors & GGUFs
└── README.md
```

---

## Quickstart & Replication

### 1. Prerequisites
* Linux or WSL2 (Ubuntu 22.04+)
* NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
* [Ollama](https://ollama.com/) installed and running
* Python 3.10+

### 2. Environment Setup
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers peft trl datasets bitsandbytes accelerate
```

### 3. Execution Order
Execute the numbered notebooks sequentially:
1. `notebooks/01_domain_adaptation.ipynb`: Trains domain weights and merges into FP16 baseline.
2. `notebooks/02_instruction_tuning.ipynb`: Performs instruction tuning and outputs `stage2_instruct/merged_model`.
3. `notebooks/03_dpo_alignment.ipynb`: Runs preference optimization against the Stage 2 baseline and saves `stage3_dpo/merged_model`.

### 4. Export & Deploy to Ollama
Once Stage 3 is completed, convert and serve the model:

```bash
python export_to_ollama.py \
    --stage-dir experiments/metformin/stage3_dpo \
    --ollama-name metformin-dpo
```

Run test queries directly in the terminal:
```bash
ollama run metformin-dpo "### Instruction: Explain how Metformin affects AMPK and hepatic gluconeogenesis. ### Response:"
```

---

## Model Evaluation: SFT vs. DPO

During Stage 2 (SFT), the model learned clinical terminology and structure, but frequently suffered from **polarity inversion** (e.g., claiming Metformin *inhibited* AMPK and *induced* gluconeogenesis).

Applying DPO in Stage 3 resolved this alignment failure:

```text
Prompt:
Explain how Metformin affects AMPK and hepatic gluconeogenesis.

Stage 2 (SFT Baseline):
"Inhibition of AMP-activated protein kinase (AMPK) induces gluconeogenesis via activation of TORC2, 
leading to the generation of 3-PGDAs."
[Result: Inverted clinical polarity; falsely asserts Metformin inhibits AMPK.]

Stage 3 (DPO Aligned):
"In vivo studies suggest that Metformin improves insulin sensitivity by regulating AMPK activity, 
which promotes GLUT4 translocation to the cell membrane..."
[Result: Hallucination eliminated; correctly captures therapeutic action and positive modulation.]
```

---

## References & Credits
* [freeCodeCamp LLM Fine-Tuning Course](https://www.freecodecamp.org/) by **Sunny Savita**
* Base model: [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
* Hugging Face [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
* [llama.cpp](https://github.com/ggerganov/llama.cpp) & [Ollama](https://github.com/ollama/ollama)