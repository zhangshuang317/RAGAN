import torch
from torch.utils import data
from dataset_brain import Dataset_gan
from RAGAN import netD, define_G
from utils import label2onehot, classification_loss, gradient_penalty, seed_torch, update_lr
import numpy as np
import time
import matplotlib as mpl
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
mpl.use('Agg')

seed_torch()

print('*******************train_gan*******************')
file_path = './brats_npy/npy_gan/gan_t1.npy'
train_data = Dataset_gan(file_path)
batch_size = 128

############fixed img
test_data = Dataset_gan('./brats_npy/npy_train/train_t1.npy')
fix_loader = data.DataLoader(dataset=test_data, batch_size=1, num_workers=4)
fix_iter = iter(fix_loader)
for i in range(40):
    next(fix_iter)
flair_fix, t1_fix, t1ce_fix, t2_fix, seg_fix = next(fix_iter)
origin_fix = np.hstack((t2_fix[0][0], flair_fix[0][0], t1ce_fix[0][0], t1_fix[0][0], seg_fix[0][0]))
for i in range(140):
    next(fix_iter)
flair_fix_2, t1_fix_2, t1ce_fix_2, t2_fix_2, seg_fix_2 = next(fix_iter)
origin_fix_2 = np.hstack((t2_fix_2[0][0], flair_fix_2[0][0], t1ce_fix_2[0][0], t1_fix_2[0][0], seg_fix_2[0][0]))
for i in range(100):
    next(fix_iter)
flair_fix_3, t1_fix_3, t1ce_fix_3, t2_fix_3, seg_fix_3 = next(fix_iter)
origin_fix_3 = np.hstack((t2_fix_3[0][0], flair_fix_3[0][0], t1ce_fix_3[0][0], t1_fix_3[0][0], seg_fix_3[0][0]))
print('------------------finsh fixed image------------------')
del fix_loader, fix_iter, test_data


'''
Parameter loading...
'''
train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

generator = define_G(4, 1, 64, 'unet', norm='instance', )
discriminator = netD()

optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

generator.cuda()
discriminator.cuda()
EPOCH = 100
num_iter = len(train_loader)
D_LOSS = []
G_LOSS = []
f = open("./loss_gan.txt", 'a')
print(time.strftime('|---------%Y-%m-%d   %H:%M:%S---------|', time.localtime(time.time())), file=f)
discriminator.train()

LAMBDA_CLS = 20
LAMBDA_REC = 0.1
LAMBDA_GP = 100

'''
Dynamic Learning Rate
'''
for epoch in range(EPOCH):
    if epoch == 30:
        update_lr(optimizer_g, 0.0001)
        update_lr(optimizer_d, 0.0001)
        print('change lr to :', optimizer_g.param_groups[0]['lr'])
    elif epoch == 60:
        update_lr(optimizer_g, 0.00005)
        update_lr(optimizer_d, 0.00005)
        print('change lr to :', optimizer_g.param_groups[0]['lr'])
    elif epoch == 90:
        update_lr(optimizer_g, 0.00001)
        update_lr(optimizer_d, 0.00001)
        print('change lr to :', optimizer_g.param_groups[0]['lr'])

    '''
    Initialization Parameters
    '''
    d_loss_ = 0
    g_loss_ = 0
    d_loss_real_ = 0
    d_loss_cls_ = 0
    d_loss_fake_ = 0
    d_loss_gp_ = 0
    g_loss_fake_ = 0
    g_loss_cls_ = 0
    g_loss_rec_ = 0
    g_loss_ragan_ = 0
    generator.train()

    '''
    Create multimodel label
    '''
    for i, (flair, t1, t1ce, t2, seg) in enumerate(train_loader):
        label_ = torch.randint(3, (t1.size(0),))
        label = label2onehot(label_, 3).cuda()
        real = torch.zeros(t1.size(0), t1.size(1), t1.size(2), t1.size(3))
        for i, l in enumerate(label_):
            if l == 0:
                real[i] = flair[i]
            elif l == 1:
                real[i] = t1ce[i]
            elif l == 2:
                real[i] = t1[i]
            else:
                print('erro!!!')

        '''
        discriminator
        '''
        out_src, out_cls = discriminator(real.float().cuda(), t2.float().cuda())
        d_loss_real = - torch.mean(out_src.sum([1, 2, 3]))#True Discrimination Loss Fuction
        d_loss_cls = classification_loss(out_cls, label)#Classification Discrimination Loss Fuction
        fake = generator(t2.float().cuda(), label)#Composite image
        out_src, out_cls = discriminator(fake.detach(), t2.float().cuda())
        d_loss_fake = torch.mean(out_src.sum([1, 2, 3]))#Fake Discrimination Loss Fuction
        alpha = torch.rand(real.size(0), 1, 1, 1).cuda()
        x_hat = (alpha * real.cuda().data + (1 - alpha) * fake.data).requires_grad_(True)
        out_src, _ = discriminator(x_hat, t2.float().cuda())
        d_loss_gp = gradient_penalty(out_src, x_hat)#Wasserstein adversarial loss and Gradient penalty loss
        d_loss =  d_loss_real +  d_loss_fake + LAMBDA_CLS * d_loss_cls + LAMBDA_GP * d_loss_gp
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        '''
        Generator
        '''
        fake = generator(t2.float().cuda(), label)#Composite image
        out_src, out_cls = discriminator(fake, t2.float().cuda())
        g_loss_fake = -torch.mean(out_src.sum([1, 2, 3]))
        g_loss_cls = classification_loss(out_cls, label)
        g_loss_rec = torch.mean(torch.abs(real.float().cuda() - fake).sum([1, 2, 3]))

        '''
        RAGAN
        '''
        fake = generator(t2.float().cuda(), label)
        fake_t2 = generator(fake.float().cuda(), label)
        g_loss_ragan = torch.mean(torch.abs(fake.float().cuda() - fake_t2).sum([1, 2, 3]))#Cycle consistency loss
        g_loss =  g_loss_fake + LAMBDA_CLS * g_loss_cls + LAMBDA_REC * g_loss_rec + LAMBDA_REC * g_loss_ragan

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        d_loss_ += d_loss.data.cuda()
        g_loss_ += g_loss.data.cuda()

        d_loss_real_ += d_loss_real.data.cuda()
        d_loss_cls_ += d_loss_cls.data.cuda()
        d_loss_fake_ += d_loss_fake.data.cuda()
        d_loss_gp_ += d_loss_gp.data.cuda()

        g_loss_fake_ += g_loss_fake.data.cuda()
        g_loss_cls_ += g_loss_cls.data.cuda()
        g_loss_rec_ += g_loss_rec.data.cuda()
        g_loss_ragan_ += g_loss_ragan.data.cuda()



    print('EPOCH %d : d_loss = %.4f , g_loss = %.4f ' % (
    epoch, d_loss_ / num_iter, g_loss_ / num_iter))
    print(
        " d_real = %.4f , d_fake = %.4f , d_cls = %.4f , d_gp = %.4f | \n "
        "g_fake = %.4f , g_cls = %.4f , g_rec = %.4f , g_rec_t2 = %.4f" % (
        d_loss_real_ / num_iter, d_loss_fake_ / num_iter, d_loss_cls_ / num_iter, d_loss_gp.data.cuda(),
        g_loss_fake_ / num_iter, g_loss_cls_ / num_iter, g_loss_rec_ / num_iter, g_loss_ragan_ / num_iter))
    print(
        "EPOCH %d : d_loss = %.4f , d_real = %.4f , d_fake = %.4f , d_cls = %.4f , d_gp = %.4f | \n"
        "g_loss = %.4f , g_fake = %.4f , g_cls = %.4f , g_rec = %.4f , g_rec_t2 = %.4f" % (
        epoch, d_loss_ / num_iter, d_loss_real_ / num_iter, d_loss_fake_ / num_iter, d_loss_cls_ / num_iter, d_loss_gp.data.cuda(),
        g_loss_ / num_iter, g_loss_fake_ / num_iter, g_loss_cls_ / num_iter, g_loss_rec_ / num_iter, g_loss_ragan_ / num_iter), file=f)
    D_LOSS.append(d_loss_ / num_iter)
    G_LOSS.append(g_loss_/ num_iter)
    x = [i for i in range(epoch + 1)]
    plt.plot(x, G_LOSS, label='generator')
    plt.plot(x, D_LOSS, label='discriminator')
    print('---------------loss plot----------------')
    plt.legend()
    plt.grid(True)
    plt.savefig('gan_bw.png', format='png')
    plt.close()


f.close()

model_save_g = './weight/generator_t2_tumor_bw.pth'
model_save_d = './weight/discriminator_t2_bw.pth'
torch.save(generator.state_dict(), model_save_g)
torch.save(discriminator.state_dict(), model_save_d)
