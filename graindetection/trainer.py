"""Trainer class for grain segmentation"""

import numpy as np
from tqdm import tqdm

from pytorchutils.globals import torch, DEVICE
from pytorchutils.basic_trainer import BasicTrainer

class Trainer(BasicTrainer):
    """Wrapper class for training routine"""
    def __init__(self, config, model, dataprocessor):
        BasicTrainer.__init__(self, config, model, dataprocessor)

    def learn_from_epoch(self):
        """Training method"""
        epoch_loss = 0
        try:
            inp_batches, out_batches = self.get_batches_fn()
        except AttributeError:
            print(
                "Error: No nb_batches_fn defined in preprocessor. "
                "This attribute is required by the training routine."
            )
        for batch_idx in tqdm(range(len(inp_batches))):
            pred_out = self.model(torch.Tensor(inp_batches[batch_idx]).to(DEVICE))
            pred_out = torch.sigmoid(pred_out)
            batch_loss = self.loss(
                pred_out,
                torch.Tensor(out_batches[batch_idx]).to(DEVICE)
            )

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()
        epoch_loss /= len(inp_batches)

        return epoch_loss

    def evaluate(self, inp, out):
        """Prediction and error estimation for given input and output"""
        with torch.no_grad():
            # Switch to PyTorch's evaluation mode.
            # Some layers, which are used for regularization, e.g., dropout or batch norm layers,
            # behave differently, i.e., are turnd off, in evaluation mode
            # to prevent influencing the prediction accuracy.
            if isinstance(self.model, (list, np.ndarray)):
                for idx, __ in enumerate(self.model):
                    self.model[idx].eval()
            else:
                self.model.eval()
            pred_out = self.model(torch.Tensor(inp).to(DEVICE))
            pred_out = torch.sigmoid(pred_out)

            # RMSE is the default accuracy metric
            error = torch.sqrt(
                self.loss(
                    pred_out,
                    torch.Tensor(out).to(DEVICE)
                )
            )
            return pred_out, (error * 100.0)
