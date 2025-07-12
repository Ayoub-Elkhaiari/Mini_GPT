import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from dataset_prep import create_dataset, DatasetPrep

def train_model(num_steps=4000, batch_size=32, lr=1e-3, block_size=64, save_path="mini_gpt_v2.pth"):

    # Prepare dataset
    math_data = create_dataset(num_samples=100)
    dataset = DatasetPrep(math_data, block_size=block_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    config = GPTConfig(vocab_size=dataset.vocab_size, block_size=block_size)
    model = GPT(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(num_steps):
        x, y = dataset.get_batch('train', batch_size=batch_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
