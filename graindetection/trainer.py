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
            batches = self.get_batches_fn()
        except AttributeError:
            print(
                "Error: No nb_batches_fn defined in preprocessor. "
                "This attribute is required by the training routine."
            )
        for batch_idx, batch in enumerate(tqdm(batches)):
            inp = batch['F']
            out = batch['T']

            pred_out = self.model(inp.to(DEVICE))
            pred_out = torch.sigmoid(pred_out)

            out_border = torch.empty(
                (out.size()[0], out.size()[-2] * out.size()[-1]),
                dtype=torch.float32
            )
            pred_border = torch.empty(
                (out.size()[0], out.size()[-2] * out.size()[-1]),
                dtype=torch.float32
            )
            for idx, image in enumerate(out):
                target_masked, pred_masked = self.preprocessor.find_border_pxl(
                    image,
                    pred_out[idx],
                    batch_idx * len(batch) + idx
                )
                out_border[idx] = torch.from_numpy(target_masked).float()
                pred_border[idx] = torch.from_numpy(pred_masked).float()

            batch_loss = self.loss(
                pred_out,
                out.to(DEVICE)
            ) + self.loss(
                pred_border.to(DEVICE),
                out_border.to(DEVICE)
            )

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()
        epoch_loss /= len(batches)

        return epoch_loss

    def evaluate(self, inp):
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

            pred_out = self.model(inp.to(DEVICE))
            pred_out = torch.sigmoid(pred_out)

            return pred_out
