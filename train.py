import torch
import os
import numpy as np
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import sys

class CustomTransform:
    def __init__(self, t_size):
        self.t_size = t_size
    def __call__(self, img):
        if img.size[0] < self.t_size or img.size[1] < self.t_size:
            img = img.resize([250, 250])
        return img


def main():
    save_dir = sys.argv[1]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    lr = 0.0001
    # transformation
    transform_train = transforms.Compose([
         CustomTransform(224),
         transforms.RandomCrop(224, padding=0),
         transforms.ToTensor()])

    # dataset 
    train_dst = torchvision.datasets.ImageFolder('./data/train', transform=transform_train)
    val_dst = torchvision.datasets.ImageFolder('./data/val', transform=transform_train)

    # dataloader
    train_dataloader = DataLoader(train_dst, batch_size=60, num_workers=10, pin_memory=True, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dst, batch_size=10, num_workers=10, pin_memory=True, shuffle=False, drop_last=False)


    # weight?? not important
    net = models.resnet50(num_classes=12, pretrained=True).cuda()
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(200):
        print 'epoch %s starting'%epoch
        net.train()
        for step, (data, target) in  enumerate(train_dataloader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            logits = net(data)

            loss = criteria(logits, target)
            optimizer.zero_grad()

            if step % 10 == 0:
                print step, "loss:", loss.data[0]
            
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9**(1e-4)
        del logits
        del loss
        del data, target
        net.eval()
        loss_list = []
        for _, (data, target) in  enumerate(val_dataloader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            logits = net(data)
            loss = criteria(logits, target)
            loss_list.append(loss.data[0])
        print 'at epoch%s, val loss is %s'%(epoch, np.mean(loss_list))
        del logits
        del loss
        del data, target
        if epoch % 8 ==0 or epoch < 10:
            print 'model %s saving'%epoch
            torch.save(net.state_dict(), os.path.join(save_dir, 'checkpoints', 'epoch%s'%epoch))
if __name__ == '__main__':
    main()
