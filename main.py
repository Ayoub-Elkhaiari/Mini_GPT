import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from dataset_prep import create_dataset, DatasetPrep

def generate(model, dataset, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0, top_k: int = None):
    model.eval()
    device = next(model.parameters()).device

    idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -dataset.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        vocab_size = logits.shape[-1]
        k = min(top_k if top_k is not None else vocab_size, vocab_size)

        if top_k is not None:
            top_logits, top_indices = torch.topk(logits, k)
            probs = F.softmax(top_logits, dim=-1)
            next_token = top_indices.gather(1, torch.multinomial(probs, num_samples=1))
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return dataset.decode(idx[0].tolist())

def main():
    toy_data = create_dataset(num_samples=100)
    dataset = DatasetPrep(toy_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GPTConfig(vocab_size=dataset.vocab_size, block_size=dataset.block_size)
    model = GPT(config).to(device)
    model.load_state_dict(torch.load("mini_gpt_v2.pth", map_location=device))

    prompt = "A train travels"
    output_text = generate(model, dataset, prompt, max_new_tokens=100, temperature=0.8, top_k=20)
    print("Generated text:\n", output_text)

if __name__ == "__main__":
    main()
