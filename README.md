# Llama-2-7b PEFT Fine-Tuning on Dolly-15k

This report contains the code and results for fine-tuning the `meta-llama/Llama-2-7b-hf` model on the `databricks/databricks-dolly-15k` dataset.

The training was performed using Parameter-Efficient Fine-Tuning (PEFT), specifically **QLoRA**, to achieve highly efficient training on NVIDIA H100 GPUs. The goal was to improve the base model's instruction-following and conversational abilities.

##  Setup

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
    cd your-repo
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

##  Model Checkpoints

The fine-tuned LoRA adapters for each experimental run are available for download. You can merge them with the base `meta-llama/Llama-2-7b-hf` model to get the final, fine-tuned `safetensors` file.

| Run Name | Output Dir | Key Hyperparameters | Google Drive Link (safetensors) |
| :--- | :--- | :--- | :--- |
| **Run 3 (Best Eval)** | `outputs2` | **Rank: 64**, LR: 1e-4, 4 Epochs | https://drive.google.com/file/d/1DWC2erPSB52tnOE6r9XaFtclSwomNwLc/view?usp=drive_link |
| **Run 2** | `outputs4` | **Rank: 64**, LR: 5e-5, 6 Epochs | https://drive.google.com/file/d/1VrybH2nkrb3FWxXltD7-KmaWFGAirFgb/view?usp=drive_link |
| **Run 4** | `outputs` | **Rank: 8**, LR: 2e-4, 3 Epochs | https://drive.google.com/file/d/1F9U85KfO2ykoO0XIKDgN2wCPVVhH23_i/view?usp=drive_link |

### Training

Conducted four main experiments, each with different hyperparameters.

### Experimental Runs

* **`outputs` (Run 4):** A baseline run with a small LoRA rank ($r=8$) and 1024 sequence length.
* **`outputs2` (Run 3):** A high-rank ($r=64$) run with a learning rate of 1e-4. **This model produced the best evaluation results on MT-Bench and MMLU.**
* **`outputs4` (Run 2):** A high-rank ($r=64$) run with a lower learning rate (5e-5) and more epochs (6) to test for convergence.
* **`outputs5` (Run 1 - DDP):** A run using **Distributed Data Parallel (DDP)** across 2x H100 GPUs. This run achieved the lowest training and validation loss.

### For viewing Visualizations
click this link: https://api.wandb.ai/links/srisona361-national-university-of-singapore/osbq1q5b 
### How to Re-Train

The training was performed using the `train2.py` script.
For SBATCH file run Submit2.sh

### For DDP implementation
Inside ddp folder both sbatch and python files are there for reproducing purpose.

