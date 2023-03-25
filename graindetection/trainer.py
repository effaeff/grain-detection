"""Trainer class for grain segmentation"""

import numpy as np
from tqdm.auto import tqdm, trange

import cv2

from pytorchutils.globals import torch, DEVICE, nn
from pytorchutils.basic_trainer import BasicTrainer
from torchmetrics import JaccardIndex


class Trainer(BasicTrainer):
    """Wrapper class for training routine"""
    def __init__(self, config, model, dataprocessor):
        BasicTrainer.__init__(self, config, model, dataprocessor)
        torch.autograd.set_detect_anomaly(True)

    def learn_from_epoch(self, epoch_idx):
        """Training method"""
        epoch_loss = 0
        try:
            batches = self.get_batches_fn()
        except AttributeError:
            print(
                "Error: No nb_batches_fn defined in preprocessor. "
                "This attribute is required by the training routine."
            )
        pbar = tqdm(batches, desc=f'Epoch: {epoch_idx}', unit='batch')
        for batch_idx, batch in enumerate(pbar):
            # batch = batches[batch_idx]
            inp = batch['F']
            out = batch['T']

            pred_out, pred_edges = self.model(inp.to(DEVICE))
            pred_out = torch.sigmoid(pred_out)
            pred_edges = torch.sigmoid(pred_edges)

            # out_border = torch.empty(out.size())
            out_border = torch.empty(
                (out.size()[0], out.size()[-2], out.size()[-1]),
                dtype=torch.float32
            )
            # pred_border = torch.empty(
                # (out.size()[0], out.size()[-2] * out.size()[-1]),
                # dtype=torch.float32
            # )
            for idx, image in enumerate(out):
                # target_masked, pred_masked = self.preprocessor.find_border_pxl(
                    # image,
                    # pred_out[idx],
                    # batch_idx * len(batch) + idx
                # )
                # out_border[idx] = torch.from_numpy(target_masked).float()
                # pred_border[idx] = torch.from_numpy(pred_masked).float()
                target = np.argmax(image.cpu().detach().numpy(), axis=0)
                # target = np.reshape(target, (out.size()[-1], out.size()[-1]))
                out_blur = cv2.GaussianBlur((target * 255).astype('uint8'), (5, 5), 0)
                target_edges = cv2.Canny(out_blur, 100, 200) / 255
                # out_border[idx] = nn.functional.one_hot(
                    # torch.from_numpy(target_edges).long()
                # ).permute(2, 0, 1).float()
                out_border[idx] = torch.from_numpy(target_edges).float()

                # pred = np.argmax(pred_out[idx].cpu().detach().numpy(), axis=0)
                # pred = np.reshape(pred, (pred_out.size()[-1], pred_out.size()[-1]))
                # pred_blur = cv2.GaussianBlur((pred * 255).astype('uint8'), (5, 5), 0)
                # pred_edges = cv2.Canny(pred_blur, 100, 200) / 255
                # pred_border[idx] = torch.from_numpy(pred_edges).float()

            batch_loss = self.loss(
                pred_out,
                out.to(DEVICE)
            ) + self.loss(
                pred_edges,
                out_border.to(DEVICE)
            )
            # batch_loss = self.loss(pred_edges, out_border.to(DEVICE))

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()

            pbar.set_postfix(batch_loss=batch_loss.item(), epoch_loss=epoch_loss/(batch_idx+1))
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

            pred_out, pred_edges = self.model(inp.to(DEVICE))
            pred_out = torch.sigmoid(pred_out)
            pred_edges = torch.sigmoid(pred_edges)

            return pred_out, pred_edges
