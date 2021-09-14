import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import shutil
import argparse
import numpy as np
import torch
from data.MNIST.data_iterator import *
from core.models.model_factory import Model
from utils import preprocess
from core import trainer
import cv2
import math

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - RST-LSTM')

# training/test
parser.add_argument('--is_training', type=int, default=1)
# parser.add_argument('--device', type=str, default='gpu:0')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_rst_lstm')
parser.add_argument('--r', type=int, default=1)
parser.add_argument('--gen_frm_dir', type=str, default='/mnt/A/meteorological/2500_ref_seq/MNIST_rst_lstm_test/')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='rst')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=2)
parser.add_argument('--layer_norm', type=int, default=1)

parser.add_argument('--non_loc_method',type = str,default='concat')# 'concat','dot_product','embedded_gauss','gauss'


# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=200)
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()
batch_size = args.batch_size


data_root = '/mnt/A/MNIST_dataset/'
train_data = MNIST(
    data_type='train',
    data_root=data_root,
)
valid_data = MNIST(
    data_type='validation',
    data_root=data_root
)
test_set1_data = MNIST(
    data_type='test_set1',
    data_root=data_root
)
test_set2_data = MNIST(
    data_type='test_set2',
    data_root=data_root
)
test_set3_data = MNIST(
    data_type='test_set3',
    data_root=data_root
)
test_set4_data = MNIST(
    data_type='test_set4',
    data_root=data_root
)
train_loader = DataLoader(train_data,
                          num_workers=2,
                          batch_size=batch_size,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=False)
valid_loader = DataLoader(valid_data,
                          num_workers=1,
                          batch_size=batch_size,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=False)
test_set1_loader = DataLoader(test_set1_data,
                         num_workers=1,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=False,
                         pin_memory=False)
test_set2_loader = DataLoader(test_set2_data,
                         num_workers=1,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=False,
                         pin_memory=False)
test_set3_loader = DataLoader(test_set3_data,
                         num_workers=1,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=False,
                         pin_memory=False)
test_set4_loader = DataLoader(test_set4_data,
                         num_workers=1,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=False,
                         pin_memory=False)




def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

def wrapper_test(model,is_save=True):
    test_save_total_root = args.gen_frm_dir
    if not os.path.exists(test_save_total_root):
        os.mkdir(test_save_total_root)
    test_save_root = os.path.join(test_save_total_root,'test_set4')
    test_loader = test_set3_loader
    if not os.path.exists(test_save_root):
        os.mkdir(test_save_root)
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    output_length = args.total_length - args.input_length
    with torch.no_grad():
        for i_batch, batch_data in enumerate(test_loader):
            print('i_batch is:', str(i_batch))
            ims = batch_data[0].numpy()
            tars = ims[:, -output_length:]
            cur_fold = batch_data[1]

            ims = preprocess.reshape_patch(ims, args.patch_size)
            img_gen,_ = model.test(ims, real_input_flag)
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            img_out = img_gen[:, -output_length:]

            # print('i_batch is:', str(i_batch), np.max(img_out), np.min(img_out))
            # print('i_batch is:', str(i_batch), np.max(tars), np.min(tars))
            img_out[img_out < 0] = 0
            mse = np.mean(np.square(img_out-tars))
            # print(mse)


            loss = loss + mse

            img_out = (img_out*255.0).astype(np.uint8)

            count = count + 1
            if is_save:
                for bat_ind in range(batch_size):
                    cur_batch_data = img_out[bat_ind,:,:,:,0]
                    cur_sample_fold = os.path.join(test_save_root,cur_fold[bat_ind])
                    if not os.path.exists(cur_sample_fold):
                        os.mkdir(cur_sample_fold)
                    for t in range(10):
                        cur_save_path = os.path.join(cur_sample_fold,'img_'+str(t+11)+'.png')
                        cur_img = cur_batch_data[t]
                        cv2.imwrite(cur_save_path, cur_img)


    print('test loss is:',str(loss/count))
    return loss / count


def wrapper_valid(model):
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))

    for i_batch, batch_data in enumerate(valid_loader):
        # print('validation batch_ind is:',str(i_batch))
        ims = batch_data.numpy()
        ims = preprocess.reshape_patch(ims, args.patch_size)
        _,mse = model.test(ims, real_input_flag)

        # mse = np.mean(np.square(tars-img_out))
        loss = loss+mse
        count = count+1

    return loss/count




def wrapper_train(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)

    eta = args.sampling_start_value
    best_mse = math.inf
    tolerate = 0
    limit = 3
    best_iter = None
    for itr in range(1, args.max_iterations + 1):
        for i_batch, batch_data in enumerate(train_loader):
            ims = batch_data.numpy()
            ims = preprocess.reshape_patch(ims, args.patch_size)
            eta, real_input_flag = schedule_sampling(eta, itr)
            cost = trainer.train(model, ims, real_input_flag, args, itr)

            if (i_batch+1) % args.display_interval == 0:
                print('itr: ' + str(itr))
                print('training loss: ' + str(cost))
        model.save(ite=itr)
        if (itr+1) % args.test_interval == 0:
            print('validation one ')
            valid_mse = wrapper_valid(model)
            print('validation mse is:',str(valid_mse))


            if valid_mse<best_mse:
                best_mse = valid_mse
                best_iter = itr
                tolerate = 0
                model.save()
            else:
                tolerate = tolerate+1

            if tolerate==limit:
                model.load()
                test_mse = wrapper_test(model)
                print('the best valid mse is:',str(best_mse))
                print('the test mse is ',str(test_mse))
                break


# if os.path.exists(args.save_dir):
#     shutil.rmtree(args.save_dir)
# os.makedirs(args.save_dir)


# gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
# args.n_gpu = len(gpu_list)
# print('Initializing models')

model = Model(args)

model.load()
# # print("the valida loss is:",str(wrapper_valid(model)))
print("the test loss is:",str(wrapper_test(model)))
# wrapper_test(model)
# if args.is_training:
#     wrapper_train(model)
# else:
#     wrapper_test(model)
