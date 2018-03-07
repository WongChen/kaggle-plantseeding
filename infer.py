import pandas as pd
import  torchvision.models as models
import numpy as np
from torch.autograd import Variable
import sys
import os
import cv2
import time
import torch
from torch.autograd import Variable
from torchvision import transforms
from testDataset import TestDataset
from torch.utils.data import DataLoader



class CustomTransform:
    def __init__(self, t_size):
        self.t_size = t_size
    def __call__(self, img):
        if img.size[0] < self.t_size or img.size[1] < self.t_size:
            img = img.resize([350, 350])
        return img


def main():
    # model_para = sys.argv[1]
    files = os.listdir('./data/test/')
    pre_dict = {}
    # net = vgg.MyVGG().cuda()
    net = models.resnet50(num_classes=12).cuda()
    net.load_state_dict(torch.load('./training/first/checkpoints/epoch184'))
    net.eval()


    classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    transform_train = transforms.Compose([
         CustomTransform(224),
         transforms.RandomCrop(224, padding=0),
         transforms.ToTensor()])
    dst = TestDataset('./data/test', [transform_train for _ in range(20)])
    dataloader = DataLoader(dst, batch_size=1, num_workers=10, pin_memory=True, shuffle=False)
    
    criteria = torch.nn.CrossEntropyLoss()
    file_list = []
    predict_list = []
    for idx, (data, name) in enumerate(dataloader):
        file_list.append(name[0])
        data = Variable(torch.cat(data).cuda())
        logits = net(data)
        probability = torch.nn.functional.softmax(logits, dim=1)
        probability = torch.mean(probability, dim=0)
        predict = probability.data.max(0)[1]
        predict = int(predict)
        predict_list.append(classes[predict])



    # df = 
    # idx = 0
    # for file_name in files:
        # idx += 1
        # print '%s image, name: %s'%(idx, file_name)
        # img = cv2.imread('./data/test/'+file_name, -1)
        # img = center_crop(img).astype(np.float32).transpose((2,0,1)).reshape(1, 3, 224, 224)
        # img = Variable(torch.from_numpy(img).cuda())
        # probability = torch.nn.functional.softmax(net(img), dim=1)
        # # probability = net(img)
        # probability = list(probability.data.cpu().numpy().reshape([120]))
        # id_name = file_name.split('.')[0]
        # probability.insert(0, id_name)

        # temp_df = pd.DataFrame([probability], columns=[id_.split('.')[0] for id_ in (['id'] + sorted(train_dataset.class_mapping.keys()))])
        # df = df.append(temp_df)
    
    df = pd.DataFrame({'file': pd.Series(file_list), 'species': pd.Series(predict_list)})
    df.to_csv('./submit/submit.csv', index=False, index_label=False)





if __name__ == '__main__':
    main()
