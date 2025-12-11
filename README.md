# Domain-Adapted RAG System: Paul Graham Essays
### GenAI Final Exam Submission

This repository contains an end-to-end solution for building a domain-specific Retrieval Augmented Generation (RAG) system. The project focuses on the Paul Graham essay corpus and demonstrates three key phases of GenAI development: pre-training custom embedding models, post-training a Small Language Model (SLM), and benchmarking the final system against frontier models.

## Project Overview

The project is divided into three core components:

1.  **Part 1: Pre-training Embeddings**
    * Trained four bi-encoder models from scratch using **BERT-base-uncased**.
    * **Architectures**: Compared **Multiple Negatives Ranking (MNR)** (Supervised) vs. **SimCSE** (Unsupervised).
    * **Data**: Compared General domain (Wikitext-103) vs. Specific domain (Paul Graham Essays).

2.  **Part 2: LLM Post-Training**
    * Fine-tuned **Google's Gemma-3-1b-it** using **LoRA** (Low-Rank Adaptation).
    * Compared three data strategies:
        * **Synthetic**: Q&A pairs generated via Llama 3/Qwen 3 (Groq API).
        * **Base**: Raw completion chunks from essays.
        * **Combined**: A hybrid dataset of both.

3.  **Part 3: RAG System & Benchmarking**
    * Built a full RAG pipeline retrieving from the custom embedding models.
    * Benchmarked generation quality (BERTScore, ROUGE-L) and retrieval accuracy.
    * **Comparison**: Evaluated local models against an **OpenAI GPT-4o-mini** baseline.

## Hardware & Requirements

**Hardware Used:**
* This project was developed and executed on **Google Colab Pro** using an **NVIDIA A100 GPU**.
* *Note: Re-running the full training pipeline (especially SimCSE and LoRA fine-tuning) requires significant VRAM (24GB+ recommended).*

**API Keys Required:**
To run the full notebook (including synthetic data generation and benchmarking), you need the following keys in your environment:
* **Groq API Key**: Required for generating synthetic Q&A pairs using Llama 3.1 and Qwen.
* **OpenAI API Key**: Required for the final benchmark comparison (GPT-4o-mini).
* **HuggingFace Token**: Required to download base models (`bert-base`, `gemma-3-1b-it`).

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/tskunz/Pre-train-post-train-and-RAG.git](https://github.com/tskunz/Pre-train-post-train-and-RAG.git)
    cd Pre-train-post-train-and-RAG
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The core logic is contained within the Jupyter Notebook: `Trevor_Kunz_GenAI_Final.ipynb`.

1.  **Setup**: Open the notebook in Google Colab or a local Jupyter environment with GPU support.
2.  **Secrets**: Ensure your API keys are set in the Colab Secrets manager or environment variables.
3.  **Execution**: Run the cells sequentially. The notebook is structured to:
    * Generate/Load Data.
    * Train Embedding Models.
    * Fine-tune Gemma.
    * Run RAG Evaluation.

## Key Results

* **Retrieval**: The **PG-Specific SimCSE** model demonstrated the highest semantic flexibility, achieving the best Average Retrieval Score (0.82) despite lower strict ranking metrics.
* **Generation**: The **Base Adapter** (trained on raw text completion) achieved the highest BERT F1 score (0.75), competitively approaching the GPT-4o-mini baseline (0.79).
* **Conclusion**: Domain adaptation on a 1.1B parameter model allows for performance within ~5% of proprietary frontier models for specialized tasks.

## License

[MIT License](LICENSE)
