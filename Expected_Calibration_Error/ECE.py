import numpy as np
import os
import pdb
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import math

def Compute_ECE_Single_CoSOD_Dataset(dir_pred, dir_gt, dataset='CoCA', method='DCFM', n_bins=10):
    """
    :param dir_pred: the directory of predictions which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param dir_gt: the directory of groundtruth which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param dataset: dataset name;
    :param method: method name;
    :param n_bins: number of bins to be used in the histogram
    :return:
    """
    GCoSOD_datasets = ['CoCA_Zero', 'CoCA_Common_40', 'CoCA_Common_60', 'CoCA_Common_80']

    print("ECE computation and plotting; Dataset: {}; Method: {}".format(dataset, method))
    bin_boundaries = torch.linspace(0, 1, steps=n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    oracle_bar = torch.linspace(0.05, 0.95, 10)
    oracle_bar[:5] = 0.0

    oracle_line = torch.linspace(0.0, 1.0, 100)

    acc = torch.zeros(n_bins)
    conf = torch.zeros(n_bins)
    bins = torch.zeros(n_bins)

    ece = 0

    class_names = sorted(os.listdir(dir_pred))

    for i in tqdm(range(len(class_names)), desc='{}'.format(dataset)):
        file_names = sorted(os.listdir(os.path.join(dir_pred, class_names[i])))

        for file_name in file_names:
            pred = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, class_names[i], file_name))).flatten())
            pred = pred / 255.0
            pred_bi = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, class_names[i], file_name))).flatten())
            pred_bi[pred_bi < 128] = 0
            pred_bi[pred_bi > 127] = 255
            if dataset in GCoSOD_datasets:
                gt = np.asarray(Image.open(os.path.join(dir_gt, class_names[i], file_name)).convert('L')).flatten()
                gt = torch.tensor(gt)
            else:
                gt = torch.tensor(np.asarray(Image.open(os.path.join(dir_gt, class_names[i], file_name))).flatten())

            confidences = torch.maximum(pred, 1.0 - pred)

            for j, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
                in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
                if len(confidences[in_bin]) > 0:
                    conf[j] = conf[j] + torch.sum(confidences[in_bin])
                bins[j] = bins[j] + len(confidences[in_bin])
                correct = pred_bi[in_bin] == gt[in_bin]
                acc[j] = acc[j] + len(correct.masked_select(correct == True))

    n_total = torch.sum(conf)

    for k in range(len(acc)):
        acc[k] = acc[k] / (bins[k] + 1e-6)
        conf[k] = conf[k] / (bins[k] + 1e-6)
        ece += (torch.abs(conf[k] - acc[k])) * (bins[k] / n_total)
        print('Bin: {}; acc: {}; conf: {}; ece: {}'.format(k, acc[k], conf[k], ece))

        if math.isnan(ece):
            pdb.set_trace()



    plt.bar(bin_lowers, oracle_bar, color='pink', width=0.1, align='edge', edgecolor='black',
            hatch='///', alpha=0.5, label='Oracle', linewidth=2)
    plt.bar(bin_lowers, acc, color='lightblue', width=0.1, align='edge', edgecolor='black', alpha=0.8,
            label=method, linewidth=2)
    plt.plot(oracle_line, oracle_line, color='r', linestyle='dashed', linewidth=1)
    plt.text(0.02, 0.8, 'ECE = {:.3f}'.format(ece), fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('{}'.format(dataset))
    plt.legend(loc='best')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    if not os.path.exists('../ECE/{}'.format(method)):
        os.makedirs('../ECE/{}'.format(method))
    plt.savefig('../ECE/{}/{}.png'.format(method, dataset), bbox_inches='tight', dpi='figure')
    # plt.show()
    plt.close()

    # acc = acc.numpy()
    # np.save('../ECE/{}/{}.npy'.format(method, dataset), acc)
    # conf = conf.numpy()
    # np.save('../ECE/{}/{}.npy'.format(method, dataset), conf)
    # bins = bins.numpy()
    # np.save('../ECE/{}/{}.npy'.format(method, dataset), bins)


def Compute_ECE_Multiple_CoSOD_Dataset(dir_pred, dir_gt, dataset='CoCA', method='DCFM', n_bins=10):
    """
    :param dir_pred: the directory of predictions which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param dir_gt: the directory of groundtruth which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param dataset: dataset name;
    :param method: method name;
    :param n_bins: number of bins to be used in the histogram
    :return:
    """
    bin_boundaries = torch.linspace(0, 1, steps=n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    oracle_bar = torch.linspace(0.05, 0.95, 10)
    oracle_bar[:5] = 0.0

    oracle_line = torch.linspace(0.0, 1.0, 100)

    acc = torch.zeros(n_bins)
    conf = torch.zeros(n_bins)
    bins = torch.zeros(n_bins)

    ece = 0

    dataset_names = ['CoCA_Common_40', 'CoCA_Common_60', 'CoCA_Common_80']

    for dataset_name in dataset_names:
        print("ECE computation and plotting; Dataset: {}; Method: {}".format(dataset_name, method))

        class_names = sorted(os.listdir(os.path.join(dir_pred, dataset_name)))
        dir_gt = './{}/GroundTruth'.format(dataset_name)

        for i in tqdm(range(len(class_names)), desc='{}'.format(dataset_name)):
            file_names = sorted(os.listdir(os.path.join(dir_pred, dataset_name, class_names[i])))

            for file_name in file_names:
                pred = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, dataset_name, class_names[i], file_name))).flatten())
                pred = pred / 255.0
                pred_bi = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, dataset_name, class_names[i], file_name))).flatten())
                pred_bi[pred_bi < 128] = 0
                pred_bi[pred_bi > 127] = 255
                # gt = torch.tensor(np.asarray(Image.open(os.path.join(dir_gt, class_names[i], file_name))).flatten())
                gt = np.asarray(Image.open(os.path.join(dir_gt, class_names[i], file_name)).convert('L')).flatten()
                gt = torch.tensor(gt)

                # print('Filename: {}; Pred Shape: {}; GT Shape: {}'.format(file_name, pred.shape, gt.shape))

                confidences = torch.maximum(pred, 1.0 - pred)

                for j, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
                    in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
                    if len(confidences[in_bin]) > 0:
                        conf[j] = conf[j] + torch.sum(confidences[in_bin])
                    bins[j] = bins[j] + len(confidences[in_bin])
                    correct = pred_bi[in_bin] == gt[in_bin]
                    acc[j] = acc[j] + len(correct.masked_select(correct == True))

    n_total = torch.sum(conf)

    for k in range(len(acc)):
        # ece += (conf[k] - acc[k]) / (conf[k] + 1e-6)
        acc[k] = acc[k] / (bins[k] + 1e-6)
        conf[k] = conf[k] / (bins[k] + 1e-6)
        ece += (torch.abs(conf[k] - acc[k])) * (bins[k] / n_total)
        print('Bin: {}; acc: {}; conf: {}; ece: {}'.format(k, acc[k], conf[k], ece))

        if math.isnan(ece):
            pdb.set_trace()



    plt.bar(bin_lowers, oracle_bar, color='pink', width=0.1, align='edge', edgecolor='black',
            hatch='///', alpha=0.5, label='Oracle', linewidth=2)
    plt.bar(bin_lowers, acc, color='lightblue', width=0.1, align='edge', edgecolor='black', alpha=0.8,
            label=method, linewidth=2)
    plt.plot(oracle_line, oracle_line, color='r', linestyle='dashed', linewidth=1)
    plt.text(0.02, 0.8, 'ECE = {:.3f}'.format(ece), fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('{}'.format(dataset))
    plt.legend(loc='best')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    if not os.path.exists('../ECE/{}'.format(method)):
        os.makedirs('../ECE/{}'.format(method))
    plt.savefig('../ECE/{}/{}.png'.format(method, dataset), bbox_inches='tight', dpi='figure')
    plt.show()
    plt.close()

    acc = acc.numpy()
    np.save('../ECE/{}/{}.npy'.format(method, dataset), acc)



if __name__ == '__main__':
    dir_pred = '../Codes/CADC/Preds/Mix_14/CoCA'
    dir_gt = './CoCA/binary'

    methods = ['DCFM', 'ICNet', 'CADC', 'GCoNet', 'GCAGC', 'GICD', 'Ours']
    # methods = ['Ours']
    # methods = ['Ours']
    # datasets = ['CoCA_Common_40', 'CoCA_Common_60', 'CoCA_Common_80', 'CoCA_Zero']
    datasets = ['CoCA_Zero']

    for dataset in datasets:
        for method in methods:
            dir_pred = '../Codes/CADC/Preds/{}/{}'.format(method, dataset)
            dir_gt = './{}/GroundTruth'.format(dataset)
            Compute_ECE_Single_CoSOD_Dataset(dir_pred=dir_pred, dir_gt=dir_gt, dataset=dataset, method=method, n_bins=10)

    # for dataset in datasets:
    #     for method in methods:
    #         dir_pred = '../Codes/CADC/Preds/{}'.format(method)
    #         dir_gt = './{}/GroundTruth'.format(dataset)
    #         compute_ECE_v3(dir_pred=dir_pred, dir_gt=dir_gt, dataset=dataset, method=method, n_bins=10)