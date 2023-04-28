"""Data processing methods"""

import re
import os
import random
from tqdm import tqdm
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
from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryJaccardIndex
from sklearn.model_selection import train_test_split

class GrainDataset(torch.utils.data.Dataset):
    """PyTorch Dataset to store grain data"""
    def __init__(self, path_features, path_target=None):
        # self.data_features = dset.ImageFolder(
            # root=path_features,
            # transform=Compose([
                # ToTensor(),
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])
        # )
        self.data_targets = None
        if path_target is not None:
            t_names = sorted(
                os.listdir(f'{path_target}/1'), key=lambda name: int(re.search('\d+', name).group())
            )
            # t_names = os.listdir(f'{path_target}/1')
            t_names = [
                t_name for t_name in t_names
                if (
                    not np.all(np.load(f'{path_target}/1/{t_name}')==0) and
                    not np.all(np.load(f'{path_target}/1/{t_name}')==1)
                )
            ]
            self.data_targets = np.array([
                np.load(f'{path_target}/1/{t_name}')
                for t_name in t_names
            ])
            # self.data_targets = dset.ImageFolder(
                # root=path_target,
                # transform=Grayscale(num_output_channels=1)
            # )

        f_names = sorted(
            os.listdir(f'{path_target}/1'), key=lambda name: int(re.search('\d+', name).group())
        )
        # f_names = os.listdir(f'{path_target}/1')
        if path_target is not None and len(f_names) != len(t_names):
            f_names = t_names

        # self.data_features = np.empty((len(f_names), 2, 256, 256))
        # for idx, f_name in enumerate(f_names):
            # print(f_name)
            # f = np.load(f'{path_features}/1/{f_name}')
            # self.data_features[idx] = f
        self.data_features = np.array([
            np.load(f'{path_features}/1/{f_name}')
            for f_name in f_names
        ])


    def __getitem__(self, index):
        # features, __ = self.data_features[index]
        features = torch.from_numpy(self.data_features[index]).float()

        targets = []
        if self.data_targets is not None:
            # targets, __ = self.data_targets[index]
            # targets = np.array(targets)
            # targets = (targets - np.min(targets)) / np.ptp(targets)

            # targets = targets.astype('uint8')
            # targets = onehot(targets, 2)
            # targets = targets.swapaxes(0, 2).swapaxes(1, 2)
            # targets = torch.FloatTensor(targets)

            targets = self.data_targets[index]
            targets = nn.functional.one_hot(
                torch.LongTensor(targets)
            ).permute(2, 0, 1).float()

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
        self.data_types = self.config['data_types']
        self.data_labels = self.config['data_labels']
        self.cscales = [
            matplotlib.cm.get_cmap(cscale_lbl)
            for cscale_lbl in self.config.get('colorscales', ['viridis' for __ in self.data_types])
        ]
        # Augmentation properties
        self.orig_size = self.config.get('orig_size', 1.0)
        desired_image_dimensions = self.config.get('processed_dim', [512, 512])
        self.height = desired_image_dimensions[0]
        self.width = desired_image_dimensions[1]

        self.process()

    def process(self):
        """Method for processing raw data into train and test data"""
        if not any(
            Path(f'{self.processed_dir}/train/{self.data_labels[-1]}/1').iterdir()
        ):
            print(any(Path(f'{self.processed_dir}/train/{self.data_labels[-1]}/1').iterdir()))
            filenames = [
                [file for file in os.listdir(self.data_dir) if file.endswith(f'{type}.txt')]
                for type in self.data_types
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
                self.data_to_images(
                    train_test[0],
                    self.data_dir,
                    f'{self.processed_dir}/{train_test[1]}'
                )

        self.train_dataset = GrainDataset(
            f'{self.processed_dir}/train/{self.data_labels[0]}',
            f'{self.processed_dir}/train/{self.data_labels[1]}'
        )
        self.test_dataset = GrainDataset(
            f'{self.processed_dir}/test/{self.data_labels[0]}',
            f'{self.processed_dir}/test/{self.data_labels[1]}'
        )

        self.train_data = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=0)
        self.test_data = DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=0)

    def data_to_images(self, filenames, from_dir, to_dir):
        """Process data files to images"""
        samples = []
        for measurements in filenames:
            print(f"Processing files: {measurements}")
            # data = np.load(f'{from_dir}/{measurements}')
            # data = np.nan_to_num(data['data'])
            data = np.moveaxis([
                np.loadtxt(
                    f'{from_dir}/{filename}',
                    delimiter=self.config['delimiter']
                )
                for filename in measurements
            ], 0, -1) # (H, W, C)



            ### Ines data test###
            # data = (data - np.min(data)) / np.ptp(data)
            # data = np.squeeze(data)
            # data = np.pad(data, ((0, 1), (0, 0)), mode='edge')
            #####################

            self.sample_subimages(
                data,
                samples,
                self.height,
                self.width,
                self.config.get('random_sampling', False)
            )
        samples = np.asarray(samples)

        for idx, sample in enumerate(samples):
            sample = np.moveaxis(sample, -1, 0) # (C, H, W)
            np.save(f'{to_dir}/{self.data_labels[0]}/1/{idx:04d}.npy', sample[:2, :, :])
            np.save(f'{to_dir}/{self.data_labels[1]}/1/{idx:04d}.npy', sample[2, :, :])
            # for im_idx, image in enumerate(sample):
            # im_to_save = Image.fromarray(np.uint8(self.cscales[0](sample) * 255))
            # im_to_save.save(
                # '{}/{}/1/{}_sample_{:03d}.png'.format(
                    # to_dir,
                    # self.data_labels[0],
                    # self.data_labels[0],
                    # idx
                # )
            # )

    def sample_subimages(self, data, result, height, width, random_sample=False):
        """Method for reading grain measurements"""
        # orig_height = np.min([np.shape(image)[0] for image in data])
        # orig_width = np.min([np.shape(image)[1] for image in data])
        # print(np.shape(data))
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

    def get_datasets(self):
        """Method to get datasets"""
        return self.train_dataset, self.test_dataset

    def validate(self, evaluate, epoch_idx, train=True):
        """Validation method which uses evaluation method from trainer"""
        # print("Start validation...")

        if train:
            Path(
                '{}/epoch{}'.format(self.results_dir, epoch_idx)
            ).mkdir(parents=True, exist_ok=True)

        pacc = []
        iou = []
        bpacc = []
        jacc = JaccardIndex(num_classes=self.output_size, task='multiclass')
        pbar = tqdm(self.test_data, desc='Validation', unit='batch')
        for batch_idx, batch in enumerate(pbar):
            inp = batch['F']
            out = batch['T']
            pred_out, pred_edges = evaluate(inp)

            for image_idx, image in enumerate(pred_out):
                inp_image = inp[image_idx].permute(1, 2, 0)
                # inp_image = (inp_image - inp_image.min()) / (inp_image.max() - inp_image.min())
                out_image = torch.argmax(out[image_idx], dim=0)
                pred_out_image = torch.argmax(image, dim=0).cpu()
                # pred_edges_image = torch.argmax(pred_edges[image_idx], dim=0).cpu()
                pred_edges_image = pred_edges[image_idx].cpu()

                # high = pred_edges_image.max() * 0.32
                # pred_edges_image[torch.where(pred_edges_image >= high)] = 1
                # pred_edges_image[torch.where(pred_edges_image < high)] = 0

                # pred_edges_image = torch.argmax(pred_edges[image_idx], dim=0).cpu()

                out_blur = cv2.GaussianBlur((out_image.numpy() * 255).astype('uint8'), (5, 5), 0)
                out_edges = cv2.Canny(out_blur, 100, 200) / 255

                # pred_blur = cv2.GaussianBlur((pred_out_image.numpy() * 255).astype('uint8'), (5, 5), 0)
                # pred_edges = cv2.Canny(pred_blur, 100, 200) / 255

                save_idx = batch_idx * self.batch_size + image_idx

                pacc.append((out_image == pred_out_image).float().mean().item() * 100.0)
                iou.append(jacc(pred_out_image, out_image) * 100.0)
                bpacc.append(self.calc_border_acc(out[image_idx], image, save_idx) * 100.0)

                # print(pred_edges_image)
                # print(out_edges)
                # quit()
                # iou_border.append(jacc_border(pred_edges_image, torch.from_numpy(out_edges)))

                # im_to_save = Image.fromarray(np.uint8(pred_out_image * 255))
                # im_to_save.save(f'{self.results_dir}/epoch{epoch_idx}/pred_{save_idx}.png')
                # plt.imsave(
                    # f'{self.results_dir}/epoch{epoch_idx}/pred_{save_idx}.png',
                    # pred_out_image,
                    # cmap='Greys'
                # )
                if train:
                    im_path = f'{self.results_dir}/epoch{epoch_idx}/pred_{save_idx:03d}'
                    Path(im_path).mkdir(parents=True, exist_ok=True)
                    self.plot_results(
                        [
                            inp_image[:, :, 0],
                            inp_image[:, :, 1],
                            pred_out_image,
                            out_image,
                            pred_edges_image,
                            out_edges
                        ],
                        ['Depth', 'Intensity', 'Prediction', 'Target', 'Pred edges', 'Target edges'],
                        f'{self.results_dir}/epoch{epoch_idx}/{save_idx:03d}.png'
                    )
                    np.save(f'{im_path}/depth.npy', inp_image[:, :, 0])
                    np.save(f'{im_path}/intensity.npy', inp_image[:, :, 1])
                    np.save(f'{im_path}/pred.npy', pred_out_image)
                    np.save(f'{im_path}/target.npy', out_image)
                    np.save(f'{im_path}/pred_edges.npy', pred_edges_image)
                    np.save(f'{im_path}/target_edges.npy', out_edges)


        return (
            np.mean([np.mean(iou), np.mean(pacc), np.mean(bpacc)]),
            np.mean([np.std(iou), np.std(pacc), np.std(bpacc)])
        )

    def infer(self, evaluate, infer_dir):
        """Inference method"""
        Path('{}/predictions'.format(infer_dir[0])).mkdir(parents=True, exist_ok=True)

        filenames = [
            [file for file in os.listdir(infer_dir[0]) if type in file]
            for type in self.data_types
        ]

        # Sort measurements numerically
        for type_idx, __ in enumerate(filenames):
            filenames[type_idx].sort(key=lambda f: int(re.sub(r'\D', '', f)))

        # Remove empty lists
        # filenames = [filename for filename in filenames if filename]

        # filenames = [["Alpha_98700_1700x629.txt"]]

        print(f"Performing inference using files:\n{filenames}")

        if not any(Path(f'{infer_dir[0]}/test/{self.data_labels[0]}/1').iterdir()):
            # self.data_to_images(filenames, infer_dir[0], f'{infer_dir[0]}/test')
            self.data_to_images(np.transpose(filenames), infer_dir[0], f'{infer_dir[0]}/test')

        target_available = any(Path(f'{infer_dir[0]}/test/{self.data_types[-1]}/1').iterdir())

        infer_dataset = GrainDataset(
            f'{infer_dir[0]}/test/{self.data_labels[0]}',
            f'{infer_dir[0]}/test/{self.data_labels[-1]}' if target_available else None
        )
        infer_data = DataLoader(infer_dataset, self.batch_size, shuffle=False, num_workers=0)

        pacc = []
        iou = []
        bpacc = []
        jacc = JaccardIndex(num_classes=self.output_size, task='multiclass')
        pbar = tqdm(infer_data, desc='Inference', unit='batch')
        for batch_idx, batch in enumerate(pbar):
            inp = batch['F']
            out = batch['T'] if target_available else None
            pred_out, pred_edges = evaluate(inp)

            for image_idx, image in enumerate(pred_out):
                save_idx = batch_idx * self.batch_size + image_idx
                im_path = f'{infer_dir[0]}/predictions/pred_{save_idx:03d}'
                Path(im_path).mkdir(parents=True, exist_ok=True)

                inp_image = inp[image_idx].permute(1, 2, 0)
                pred_out_image = torch.argmax(image, dim=0).cpu()
                pred_edges_image = pred_edges[image_idx].cpu()

                if out is not None:
                    out_image = torch.argmax(out[image_idx], dim=0)
                    out_blur = cv2.GaussianBlur((out_image.numpy() * 255).astype('uint8'), (5, 5), 0)
                    out_edges = cv2.Canny(out_blur, 100, 200) / 255

                    pacc.append((out_image == pred_out_image).float().mean().item() * 100.0)
                    iou.append(jacc(pred_out_image, out_image) * 100.0)
                    bpacc.append(self.calc_border_acc(out[image_idx], image, save_idx) * 100.0)

                    self.plot_results(
                        [
                            inp_image[:, :, 0],
                            inp_image[:, :, 1],
                            pred_out_image,
                            out_image,
                            pred_edges_image,
                            out_edges
                        ],
                        ['Depth', 'Intensity', 'Prediction', 'Target', 'Pred edges', 'Target edges'],
                        f'{infer_dir[0]}/predictions/{save_idx:03d}.png'
                    )
                    np.save(f'{im_path}/target.npy', out_image)
                    np.save(f'{im_path}/target_edges.npy', out_edges)
                else:
                    self.plot_results(
                        [
                            inp_image[:, :, 0],
                            inp_image[:, :, 1],
                            pred_out_image,
                            pred_edges_image,
                        ],
                        ['Depth', 'Intensity', 'Prediction', 'Pred edges'],
                        f'{infer_dir[0]}/predictions/{save_idx:03d}.png'
                    )

                np.save(f'{im_path}/depth.npy', inp_image[:, :, 0])
                np.save(f'{im_path}/intensity.npy', inp_image[:, :, 1])
                np.save(f'{im_path}/pred.npy', pred_out_image)
                np.save(f'{im_path}/pred_edges.npy', pred_edges_image)

        if pacc:
            print(
                f'pacc: {np.mean(pacc):.2f} +- {np.std(pacc):.2f}\t'
                f'iou: {np.mean(iou):.2f} +- {np.std(iou):.2f}\t'
                f'bpacc: {np.mean(bpacc):.2f} +- {np.std(bpacc):.2f}'
            )

    def find_border_pxl(self, target, output, idx=0):
        target = [np.argmax(target.cpu().detach().numpy(), axis=0)]
        target = np.reshape(target, (self.height, self.width))
        plt.imsave(f'{self.results_dir}/{idx}_target.png', target)
        target = target.flatten()

        img = cv2.imread(f'{self.results_dir}/{idx}_target.png')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = np.where(img_gray > 130, 30, 215)
        img_gray = img_gray.astype(np.uint8)
        ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, 2, 1)
        bmask = np.ones(img.shape[:2], dtype="uint8")
        cv2.drawContours(bmask, contours, -1, 0, 5)

        bsave = Image.fromarray(np.uint8(bmask * 255))
        # bsave.save(f"border_{idx}.png")

        bmask = bmask.flatten()
        output = [np.argmax(output.cpu().detach().numpy(), axis = 0)]
        pred = np.reshape(output, (self.height, self.width))
        pred = pred.flatten()

        pred_masked = np.ma.masked_where(bmask==1, pred)
        target_masked = np.ma.masked_where(bmask==1, target)

        return target_masked, pred_masked

    def calc_border_acc(self, target, output, idx=0):
        target_masked, pred_masked = self.find_border_pxl(target, output, idx)

        pred_masked = np.ma.compressed(pred_masked)
        target_masked = np.ma.compressed(target_masked)

        loss = np.sum(pred_masked==target_masked) / len(pred_masked)
        return loss

    def plot_results(self, data, titles, filename):
        """Plot prediction results"""
        __, axs = plt.subplots(1, len(data), sharey=True)
        for idx, image in enumerate(data):
            axs[idx].imshow(image, cmap='inferno')
            axs[idx].set_title(titles[idx])
        plt.savefig(
            filename,
            format='png',
            dpi=600,
            bbox_inches='tight'
        )
        plt.close()
