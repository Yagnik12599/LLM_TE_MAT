import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import Dataset
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set reproducibility seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load and process dataset
path = './alpaca_style_dataset.jsonl'
with open(path) as f:
    lines = [json.loads(line) for line in f]

df = pd.DataFrame(lines).dropna()
df = df.rename(columns={"input": "text", "output": "label"})
df["label"] = df["label"].astype(float)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

hf_dataset = Dataset.from_pandas(df[["text", "label"]])
hf_dataset = hf_dataset.map(tokenize, batched=True)
hf_dataset = hf_dataset.rename_column("label", "labels")
hf_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

class TransformerRegressor(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.embeddings.position_embeddings = nn.Embedding(
            self.encoder.config.max_position_embeddings,
            self.encoder.config.hidden_size
        )
        nn.init.zeros_(self.encoder.embeddings.position_embeddings.weight)
        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = output.last_hidden_state[:, 0, :]
        preds = self.regressor(cls_token).squeeze(-1)
        loss = nn.MSELoss()(preds, labels) if labels is not None else None
        return {"loss": loss, "logits": preds}

# Prepare dataset once (constant for all seeds)
hf_dataset_py = hf_dataset.with_format("python")
input_ids = torch.tensor([row["input_ids"] for row in hf_dataset_py])
attention_mask = torch.tensor([row["attention_mask"] for row in hf_dataset_py])
labels = torch.tensor([row["labels"] for row in hf_dataset_py], dtype=torch.float)
full_dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_train_losses = []
all_val_losses = []

all_train_losses_dict = {}
all_val_losses_dict = {}
# Try multiple seeds and track best one + store loss curves
best_seed = None
best_loss = float("inf")
with open("seeds_used.txt") as f:
    seeds = [int(line.strip()) for line in f]
all_models_info = []

for seed in seeds:
    set_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20)

    model = TransformerRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    train_losses = []
    val_losses = []

    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output["loss"].backward()
            optimizer.step()
            total_loss += output["loss"].item()

        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += output["loss"].item()
        val_losses.append(val_loss / len(val_loader))

    final_val_loss = val_losses[-1]
    print(f"Seed {seed} → Final Val Loss: {final_val_loss:.4f}")

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    
    # Save losses for this seed
    all_train_losses_dict[seed] = train_losses
    all_val_losses_dict[seed] = val_losses
    
    if final_val_loss < best_loss:
        best_loss = final_val_loss
        best_seed = seed
        best_model = model
        best_train_loader = train_loader
        best_val_loader = val_loader
    # Evaluate and store true vs preds for this seed
    model.eval()
    train_preds, train_trues, val_preds, val_trues = [], [], [], []
    
    with torch.no_grad():
        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            train_preds.extend(output["logits"].cpu().numpy())
            train_trues.extend(labels.cpu().numpy())
    
        for batch in val_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            val_preds.extend(output["logits"].cpu().numpy())
            val_trues.extend(labels.cpu().numpy())
    
    # Convert to numpy
    train_preds = np.array(train_preds)
    train_trues = np.array(train_trues)
    val_preds = np.array(val_preds)
    val_trues = np.array(val_trues)
    
    # Compute metrics
    train_mse = mean_squared_error(train_trues, train_preds)
    train_mae = mean_absolute_error(train_trues, train_preds)
    train_r2 = r2_score(train_trues, train_preds)
    
    val_mse = mean_squared_error(val_trues, val_preds)
    val_mae = mean_absolute_error(val_trues, val_preds)
    val_r2 = r2_score(val_trues, val_preds)
    
    # Append to metrics list
    all_models_info.append({
        'seed': seed,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_mse': val_mse,
        'val_mae': val_mae,
        'val_r2': val_r2
    })
    
    # Save predictions
    if 'all_preds_data' not in locals():
        all_preds_data = {}
    
    all_preds_data[seed] = {
        'train_trues': train_trues,
        'train_preds': train_preds,
        'val_trues': val_trues,
        'val_preds': val_preds
    }

print(f"\n Best seed: {best_seed} with Validation Loss = {best_loss:.4f}")
np.savez_compressed("all_seed_losses.npz", train_losses=all_train_losses_dict,
                    val_losses=all_val_losses_dict)
# Save the best model
torch.save(best_model.state_dict(), "regression_model_best_seed.pt")
# Save all metrics
metrics_df = pd.DataFrame(all_models_info)
metrics_df.to_csv("bert_seed_metrics.csv", index=False)

# Save all predictions
with open("bert_seed_true_pred.pkl", "wb") as f:
    pickle.dump(all_preds_data, f)

# Plot average ± std loss curves
mean_train = np.mean(all_train_losses, axis=0)
std_train = np.std(all_train_losses, axis=0)
mean_val = np.mean(all_val_losses, axis=0)
std_val = np.std(all_val_losses, axis=0)

epochs = range(1, 51)
plt.figure(figsize=(8,6))
plt.plot(epochs, mean_train, label="Train Loss (mean)")
plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.2)
plt.plot(epochs, mean_val, label="Val Loss (mean)")
plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (Mean ± Std)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_mean_std_curve.png")
#plt.show()

# True vs Predicted for best seed
best_model.eval()
train_preds, train_trues, val_preds, val_trues = [], [], [], []

with torch.no_grad():
    for batch in best_val_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        output = best_model(input_ids=input_ids, attention_mask=attention_mask)
        val_preds.extend(output["logits"].cpu().numpy())
        val_trues.extend(labels.cpu().numpy())

    for batch in best_train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        output = best_model(input_ids=input_ids, attention_mask=attention_mask)
        train_preds.extend(output["logits"].cpu().numpy())
        train_trues.extend(labels.cpu().numpy())

train_preds = torch.tensor(train_preds)
train_trues = torch.tensor(train_trues)
val_preds = torch.tensor(val_preds)
val_trues = torch.tensor(val_trues)

plt.figure(figsize=(8,6))
plt.scatter(train_trues, train_preds, alpha=0.7, label='Train', color='blue', edgecolor='k')
plt.scatter(val_trues, val_preds, alpha=0.7, label='Validation', color='green', edgecolor='k')
plt.plot([train_trues.min(), train_trues.max()], [train_trues.min(), train_trues.max()], 'k--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"True vs Predicted (Best Seed = {best_seed})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("true_vs_predicted_best_seed.png")
# plt.show()