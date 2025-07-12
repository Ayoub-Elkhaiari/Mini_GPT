import random
import torch

def generate_math_problem():
    problem_type = random.randint(1, 8)

    if problem_type == 1:
        name = random.choice(['John', 'Sara', 'Alex'])
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        item = random.choice(['apples', 'candies'])
        return f"{name} has {a} {item}. They get {b} more. How many do they have now? Answer: {a + b}."

    elif problem_type == 2:
        v = random.randint(30, 100)
        t = random.randint(1, 5)
        return f"A car travels at {v} km/h for {t} hours. How far does it go? Answer: {v * t} km."

    elif problem_type == 3:
        w = random.randint(2, 10)
        l = random.randint(2, 10)
        return f"A rectangle has width {w} and length {l}. What is the area? Answer: {w * l}."

    elif problem_type == 4:
        p = random.randint(1000, 5000)
        s = random.randint(100, 900)
        return f"A person has ${p} and spends ${s}. How much is left? Answer: ${p - s}."

    elif problem_type == 5:
        r = random.randint(5, 20)
        h = random.randint(1, 10)
        return f"A machine produces {r} units/hour. How many in {h} hours? Answer: {r * h} units."

    elif problem_type == 6:
        distance = random.randint(50, 200)
        return f"A train moves {distance} km in 2 hours. What is the speed? Answer: {distance // 2} km/h."

    elif problem_type == 7:
        side1 = random.randint(3, 15)
        side2 = random.randint(3, 15)
        return f"Rectangle with sides {side1} and {side2}. Perimeter? Answer: {2 * (side1 + side2)}."

    else:  # problem_type == 8
        total = random.randint(20, 50)
        given = random.randint(5, 20)
        return f"You have {total} marbles and give away {given}. How many left? Answer: {total - given}."

def create_dataset(num_samples=100):
    toy_data = [generate_math_problem() for _ in range(num_samples)]
    return toy_data

class DatasetPrep:
    def __init__(self, toy_data, block_size=64):
        self.block_size = block_size
        all_text = "\n".join(toy_data)
        self.chars = sorted(list(set(all_text)))  # includes '\n'
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        full_text = "\n".join(toy_data)
        data = self.encode(full_text)
        n = int(0.9 * len(data))
        self.train_data = torch.tensor(data[:n])
        self.val_data = torch.tensor(data[n:])

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split, batch_size=32):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y
