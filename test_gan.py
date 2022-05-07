import torch
from torch.utils import data
from dataset_brain import Dataset_gan
from model import define_G
from utils import seed_torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

seed_torch(10)

print('*******************test_gan*******************')
file_path='./brats_npy/npy_test/test_t1.npy'
model_path='./weight/generator_t2_tumor_bw.pth'
train_data=Dataset_gan(file_path)
length=len(train_data)
print('len',length)
batch_size=128
train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,num_workers=4)
gen = define_G(4, 1, 64, 'unet', norm='instance', )
gen.load_state_dict(torch.load(model_path))
gen.eval()
gen.cuda()

flair_all=np.zeros((length,128,128))
t1ce_all=np.zeros((length,128,128))
t1_all=np.zeros((length,128,128))
num_iter=len(train_loader)
start=0

with torch.no_grad():
    for i,(flair,t1,t1ce,t2,_) in enumerate(train_loader):
        ####discriminator
        c=torch.zeros(t1.size(0),3).cuda()
        c[np.arange(t1.size(0)),0]=1
        flair_pred=gen(t2.float().cuda(),c).squeeze().data.cpu().numpy()
        flair_all[start:start+t1.size(0)]=flair_pred

        c=torch.zeros(t1.size(0),3).cuda()
        c[np.arange(t1.size(0)),1]=1
        t1ce_pred=gen(t2.float().cuda(),c).squeeze().data.cpu().numpy()
        t1ce_all[start:start+t1.size(0)]=t1ce_pred

        c=torch.zeros(t1.size(0),3).cuda()
        c[np.arange(t1.size(0)),2]=1
        t1_pred=gen(t2.float().cuda(),c).squeeze().data.cpu().numpy()
        t1_all[start:start+t1.size(0)]=t1_pred

        start=start+t1.size(0)

print('******* end *******')
np.save('./brats_npy/npy_pred/pred_flair.npy',flair_all)
np.save('./brats_npy/npy_pred/pred_t1ce.npy',t1ce_all)
np.save('./brats_npy/npy_pred/pred_t1.npy',t1_all)