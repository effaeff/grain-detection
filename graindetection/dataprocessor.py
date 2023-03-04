"""Data processing methods"""

import re
import os
import random
from pathlib import Path

import cv2

import numpy as np
import torchvision.datasets as dset
import matplotlib.cm
import matplotlib.pyplot as plt

from pytorchutils.globals import nn
from pytorchutils.globals import torch

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Grayscale, Normalize, Compose
from sklearn.model_selection import train_test_split

from graindetection.dataaugmentor import augment_images


def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

class GrainDataset(torch.utils.data.Dataset):
    """PyTorch Dataset to store grain data"""
    def __init__(self, path_features, path_target=None):
        self.data_features = dset.ImageFolder(
            root=path_features,
            transform=Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        self.data_targets = None
        if path_target is not None:
            self.data_targets = dset.ImageFolder(
                root=path_target,
                transform=Grayscale(num_output_channels=1)
            )

    def __getitem__(self, index):
        features, __ = self.data_features[index]

        targets = []
        if self.data_targets is not None:
            targets, __ = self.data_targets[index]
            targets = np.array(targets)
            targets = (targets - np.min(targets)) / np.ptp(targets)

            targets = targets.astype('uint8')
            targets = onehot(targets, 2)
            targets = targets.swapaxes(0, 2).swapaxes(1, 2)
            targets = torch.FloatTensor(targets)

            # targets = nn.functional.one_hot(torch.LongTensor(targets)).permute(2, 0, 1).float()

        item = {'F': features, 'T': targets}
        return item

    def __len__(self):
        return len(self.data_features) # Assume that both datasets have equal length

class DataProcessor():
    """Class for data processor"""
    def __init__(self, config):
        self.config = config
        self.random_seed = self.config.get('random_seed', 1234)
        # Assume binary segmentation as default
        self.output_size = self.config.get('output_size', 2)
        self.batch_size = self.config.get('batch_size', 4)
        self.data_dir = self.config['data_dir']
        self.processed_dir = self.config['processed_dir']
        self.results_dir = self.config['results_dir']
        self.data_lbls = self.config['data_labels']
        self.cscales = [
            matplotlib.cm.get_cmap(cscale_lbl)
            for cscale_lbl in self.config.get('colorscales', ['viridis' for __ in self.data_lbls])
        ]
        # Augmentation properties
        self.orig_size = self.config.get('orig_size', 1.0)
        desired_image_dimensions = self.config.get('processed_dim', [512, 512])
        self.height = desired_image_dimensions[0]
        self.width = desired_image_dimensions[1]

        self.process()

    def process(self):
        """Method for processing raw data into train and test data"""
        filenames = [
            [file for file in os.listdir(self.data_dir) if file.endswith(f'{label}.txt')]
            for label in self.data_lbls
        ]

        # Sort measurements numerically
        for type_idx, __ in enumerate(filenames):
            filenames[type_idx].sort(key=lambda f: int(re.sub(r'\D', '', f)))

        # Do train/test-split
        train_size = self.config.get('train_size', 1.0) / len(np.transpose(filenames))
        test_size = self.config.get('test_size', 0.2) / len(np.transpose(filenames))
        train_files, test_files = train_test_split(
            np.transpose(filenames),
            test_size=test_size,
            random_state=self.random_seed
        )

        if train_size + test_size < 1.0:
            train_files, __ = train_test_split(
                train_files,
                train_size=self.config['train_size'] / len(train_files),
                random_state=self.random_seed
            )

        print(f"Train files:\n{train_files}")
        print(f"Test files:\n{test_files}")

        for train_test in tuple(zip([train_files, test_files], ['train', 'test'])):
            if not any(
                Path(f'{self.processed_dir}/{train_test[1]}/{self.data_lbls[0]}/1').iterdir()
            ):
                self.data_to_images(
                    train_test[0],
                    self.data_dir,
                    f'{self.processed_dir}/{train_test[1]}'
                )

        train_dataset = GrainDataset(
            f'{self.processed_dir}/train/{self.data_lbls[0]}',
            f'{self.processed_dir}/train/{self.data_lbls[1]}'
        )
        test_dataset = GrainDataset(
            f'{self.processed_dir}/test/{self.data_lbls[0]}',
            f'{self.processed_dir}/test/{self.data_lbls[1]}'
        )

        self.train_data = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=0)
        self.test_data = DataLoader(test_dataset, self.batch_size, shuffle=False, num_workers=0)

    def data_to_images(self, filenames, from_dir, to_dir):
        """Process data files to images"""
        samples = []
        for measurements in filenames:
            print(f"Processing files: {measurements}")
            # data = np.load(f'{from_dir}/{measurements}')
            # data = np.nan_to_num(data['data'])
            data = [
                np.loadtxt(
                    f'{from_dir}/{filename}',
                    delimiter=self.config['delimiter']
                )
                for filename in measurements
            ]
            data = (data - np.min(data)) / np.ptp(data)
            data = np.squeeze(data)
            data = np.pad(data, ((0, 1), (0, 0)), mode='edge')

            self.sample_subimages(
                data,
                samples,
                self.height,
                self.width,
                self.config.get('random_sampling', False)
            )
        samples = np.asarray(samples)

        for idx, sample in enumerate(samples):
            # for im_idx, image in enumerate(sample):
            im_to_save = Image.fromarray(np.uint8(self.cscales[0](sample) * 255))
            im_to_save.save(
                '{}/{}/1/{}_sample_{:03d}.png'.format(
                    to_dir,
                    self.data_lbls[0],
                    self.data_lbls[0],
                    idx
                )
            )

    def sample_subimages(self, data, result, height, width, random_sample=False):
        """Method for reading grain measurements"""
        # orig_height = np.min([np.shape(image)[0] for image in data])
        # orig_width = np.min([np.shape(image)[1] for image in data])
        print(np.shape(data))
        orig_height = np.shape(data)[0]
        orig_width = np.shape(data)[1]
        n_samples = (
            self.config['n_samples'] if random_sample else
            int(orig_height / height) * int(orig_width / width)
        )
        for idx in range(n_samples):
            sample_height = int(idx / int(orig_width / width)) * height
            sample_width = idx % int(orig_width / width) * width

            if random_sample:
                sample_height = random.randint(0, orig_height - height)
                sample_width = random.randint(0, orig_width - width)
            # samples = np.empty((len(data), height, width))
            # for im_idx, image in enumerate(data):
                # image = (image - np.min(image)) / np.ptp(image)
                # samples[im_idx] = image[
                    # sample_height:sample_height + height,
                    # sample_width:sample_width + width
                # ]
            # result.append(samples)
            result.append(data[sample_height:sample_height + height, sample_width:sample_width + width])

    def get_batches(self):
        """Method to pass batches to trainer"""
        return self.train_data

    def validate(self, evaluate, epoch_idx):
        """Validation method which uses evaluation method from trainer"""
        print("Start validation...")

        Path(
            '{}/epoch{}'.format(self.results_dir, epoch_idx)
        ).mkdir(parents=True, exist_ok=True)

        acc = []
        for batch_idx, batch in enumerate(self.test_data):
            inp = batch['F']
            out = batch['T']
            pred_out = evaluate(inp)

            for image_idx, image in enumerate(pred_out):
                inp_image = inp[image_idx].permute(1, 2, 0)
                inp_image = (inp_image - inp_image.min()) / (inp_image.max() - inp_image.min())
                out_image = torch.argmax(out[image_idx], dim=0)
                pred_out_image = torch.argmax(image, dim=0).cpu()

                save_idx = batch_idx * self.batch_size + image_idx

                # acc.append((out_image == pred_out_image).float().mean().item() * 100.0)
                acc.append(self.calc_border_acc(out[image_idx], image, save_idx) * 100.0)

                save_idx = batch_idx * self.batch_size + image_idx
                # im_to_save = Image.fromarray(np.uint8(pred_out_image * 255))
                # im_to_save.save(f'{self.results_dir}/epoch{epoch_idx}/pred_{save_idx}.png')
                plt.imsave(f'{self.results_dir}/epoch{epoch_idx}/pred_{save_idx}.png', pred_out_image, cmap='Greys')
                # self.plot_results(
                    # [inp_image, pred_out_image, out_image],
                    # ['Input', 'Output', 'Target'],
                    # f'{self.results_dir}/epoch{epoch_idx}/pred_{save_idx}.png'
                # )

        return np.mean(acc), np.std(acc)

    def infer(self, evaluate, infer_dir):
        """Inference method"""
        Path('{}/predictions'.format(infer_dir[0])).mkdir(parents=True, exist_ok=True)

        # filenames = [
            # [file for file in os.listdir(infer_dir[0]) if file.endswith(f'{label}.txt')]
            # for label in self.data_lbls
        # ]

        # Sort measurements numerically
        # for type_idx, __ in enumerate(filenames):
            # filenames[type_idx].sort(key=lambda f: int(re.sub(r'\D', '', f)))

        # Remove empty lists
        # filenames = [filename for filename in filenames if filename]

        filenames = [["Alpha_98700_1700x629.txt"]]

        print(f"Performing inference using files:\n{filenames}")

        if not any(Path(f'{infer_dir[0]}/test/{self.data_lbls[0]}/1').iterdir()):
            self.data_to_images(filenames, infer_dir[0], f'{infer_dir[0]}/test')
            # self.data_to_images(np.transpose(filenames), infer_dir[0], f'{infer_dir[0]}/test')

        target_available = any(Path(f'{infer_dir[0]}/test/{self.data_lbls[-1]}/1').iterdir())

        infer_dataset = GrainDataset(
            f'{infer_dir[0]}/test/{self.data_lbls[0]}',
            f'{infer_dir[0]}/test/{self.data_lbls[-1]}' if target_available else None
        )
        infer_data = DataLoader(infer_dataset, self.batch_size, shuffle=False, num_workers=0)

        acc = []
        bacc = []
        for batch_idx, batch in enumerate(infer_data):
            inp = batch['F']
            out = batch['T'] if target_available else None
            pred_out = evaluate(inp)
            for image_idx, image in enumerate(pred_out):
                inp_image = inp[image_idx].permute(1, 2, 0)
                inp_image = (inp_image - inp_image.min()) / (inp_image.max() - inp_image.min())
                pred_out_image = torch.argmax(image, dim=0).cpu()
                save_idx = batch_idx * self.batch_size + image_idx

                if out is not None:
                    out_image = torch.argmax(out[image_idx], dim=0)
                    acc.append((out_image == pred_out_image).float().mean().item() * 100.0)
                    bacc.append(self.calc_border_acc(out[image_idx], image, save_idx) * 100.0)

                    self.plot_results(
                        [inp_image, pred_out_image, out_image],
                        ['Input', 'Output', 'Target'],
                        f'{infer_dir[0]}/predictions/pred_{save_idx}.png',
                    )
                else:
                    plt.imsave(
                        f'{infer_dir[0]}/predictions/pred_{save_idx}.png',
                        pred_out_image,
                        cmap='Greys'
                    )
                    # im_to_save = Image.fromarray(np.uint8(pred_out_image * 255))
                    # im_to_save.save(f'{infer_dir[0]}/predictions/pred_{save_idx}.png')
        if acc:
            print(f"Accuracy: {np.mean(acc)} +- {np.std(acc)}")
            print(f"Boundary accuracy: {np.mean(bacc)} +- {np.std(bacc)}")

    def calc_border_acc(self, target, output, idx=0):
        target = [np.argmax(target.cpu().detach().numpy(), axis = 0)]
        target = np.reshape(target, (512,512))
        target = target.flatten()
        plt.imsave('target.png', np.reshape(target, (512,512)))

        img = cv2.imread('target.png')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = np.where(img_gray >130, 30, 215)
        img_gray = img_gray.astype(np.uint8)
        ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, 2, 1)
        bmask = np.ones(img.shape[:2], dtype="uint8")
        #cv2.drawContours(bmask, contours, -1, 0, 15)
        cv2.drawContours(bmask, contours, -1, 0, 15)
        #plt.imsave('{}bmask.png'.format(i), bmask, cmap='gray')

        bsave = Image.fromarray(np.uint8(bmask * 255))
        bsave.save(f"border_{idx}.png")

        bmask = bmask.flatten()
        output = [np.argmax(output.cpu().detach().numpy(), axis = 0)]
        #plt.imsave('{}output.png'.format(i), np.reshape(output, (512,512)))
        pred = np.reshape(output, (512,512))
        pred = pred.flatten()

        pred_masked = np.ma.masked_where(bmask==1, pred)
        pred_masked = np.ma.compressed(pred_masked)
        target_masked = np.ma.masked_where(bmask==1, target)
        target_masked = np.ma.compressed(target_masked)

        loss = np.sum(pred_masked==target_masked) / len(pred_masked)
        return loss

    def plot_results(self, data, titles, filename):
        """Plot prediction results"""
        __, axs = plt.subplots(1, len(data), sharey=True)
        for idx, image in enumerate(data):
            axs[idx].imshow(image)
            axs[idx].set_title(titles[idx])
        plt.savefig(
            filename,
            format='png',
            dpi=600,
            bbox_inches='tight'
        )
        plt.close()
