"""
    date:       2021/3/20 4:03 下午
    written by: neonleexiang
"""
import os
import torch
import copy
import cudnn
from pytorch_model import pytorch_SRCNN
from test_pytorch_datasets import TestDataset, TrainDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def calc_psnr(img1, img2):
    return 10. * torch.log10(255 * 255 / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_and_test_pytorch_srcnn(
    data_size=100000,
    test_data_size=1000,
    num_channels=1,
    learning_rate=0.0001,
    batch_size=64,
    num_epochs=500,
    num_workers=8,
    is_training=True,
    model_save_path='pytorch_models/'):

    if not os.path.exists(model_save_path):
        print('making dir')
        os.makedirs(model_save_path)

    print('dir: %s exits' % model_save_path)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device setting successfully')
    model = pytorch_SRCNN().to(device)
    criterion = torch.nn.MSELoss()
    """
        根据论文，我们有设定前两层的学习率为 0.0001 最后一层的学习率为0.00001
    """
    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': learning_rate * 0.1}
    ], lr=learning_rate)

    print('loading data....')
    train_dataset = TrainDataset(data_size)
    # TODO: to understand what is pin_memory and num_workers and drop_last
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = TestDataset(test_data_size)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    print('data size: %d\t%d\t' % (data_size, test_data_size))
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(num_epochs):
        # print('------> {}th epoch is training......'.format(epoch))
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epochs-1))
            for data in train_dataloader:
                images, labels = data
                # print(images.shape)
                # print(images.shape)
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(images))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(images))

        # torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(images).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(labels*255., preds*255.), len(images))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(model_save_path, 'best.pth'))


if __name__ == '__main__':
    train_and_test_pytorch_srcnn()
