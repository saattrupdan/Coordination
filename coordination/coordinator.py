import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import random

class Coordinator(nn.Module):
    def __init__(self, distance_matrix: torch.FloatTensor, dim: int = 2,
        lr: float = 3e-2):
        super().__init__()
        self.distance_matrix = torch.FloatTensor(distance_matrix)
        self.coords = nn.Parameter(torch.rand(distance_matrix.shape[0], dim))
        self.optim = optim.AdamW([self.coords], lr = lr)
        self.patience = min(100, 10 * self.distance_matrix.shape[0])
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.optim, 
            mode = 'min', patience = self.patience, factor = 0.5)
        self.goal = (self.distance_matrix.mean() / 100) ** 2

    @staticmethod
    def get_distance_matrix(X: torch.FloatTensor) -> torch.FloatTensor:
        return torch.norm(torch.sub(X[:, None], X), dim = -1)

    def get_coords(self):
        with torch.no_grad():
            self.coords -= self.coords.mean()
            self.coords /= self.coords.norm()
            return self.coords.detach().numpy()

    def compile(self, nepochs: int = 200, batch_size: int = 256):
        N: int = self.coords.shape[0]
        ema: float = 0.
        ema_factor: float = min(N, batch_size) / (10 * N)
        idxs: np.array = np.arange(N)
        nbatches: int = N // batch_size
        nbatches = nbatches + 1 if N % batch_size else nbatches
        niter: int = 0
        bad_scores: int = 0
        best_score: float = float('inf')
        finished: bool = False

        with tqdm(desc = 'Computing coordinates', total = N * nepochs) as pbar:
            for epoch in range(nepochs):
                if finished: break
                self.optim.zero_grad()
                np.random.shuffle(idxs)

                for batch in np.array_split(idxs, nbatches):
                    niter += batch.shape[0]
                    pred = self.get_distance_matrix(self.coords[batch, :])
                    true = self.distance_matrix[np.ix_(batch, batch)]
                    loss = F.mse_loss(pred, true)

                    ema = (1 - ema_factor) * ema + ema_factor * float(loss)
                    ema /= 1 - (1 - ema_factor) ** niter

                    pbar.set_description(f'Computing coordinates - '\
                                         f'mean squared error {ema:.6f}')
                    pbar.update(batch.shape[0])

                    loss.backward()
                    self.optim.step()

                    if niter > batch_size / ema_factor:
                        if ema < best_score - self.goal:
                            best_score = ema
                            bad_scores = 0
                        else:
                            bad_scores += 1

                        if bad_scores > 3 * self.patience or ema < self.goal: 
                            finished = True
                            break

                    self.sched.step(ema)

        return self

    def plot(self):
        if self.coords.shape[1] > 2:
            raise RuntimeError('Cannot plot coordinates in dimensions > 2')
        else:
            raise NotImplementedError('Cannot plot yet!')

    def score(self):
        with torch.no_grad():
            pred = self.get_distance_matrix(self.coords)
            diff = torch.mean(torch.abs(self.distance_matrix - pred))
            return -torch.log(diff).item()

if __name__ == '__main__':
    coords = torch.randn(2000, 2)
    distance_matrix = torch.norm(coords[:, None] - coords, dim = -1)

    coordinator = Coordinator(distance_matrix, dim = 2).compile()
    print(coordinator.score())
