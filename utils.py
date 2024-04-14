import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import h5py                                           #import h5 files
import os                                             #OS operations
import time                                           #timing and clock time
import datetime                                       #date and time
import scipy.signal as signal                         #signal processing
from scipy.io import loadmat                          #load MatLab m-files
from scipy.stats.qmc import LatinHypercube            #Latin Hypercube sampling

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as image_ssim
from skimage.metrics import mean_squared_error as image_mse
from skimage.metrics import peak_signal_noise_ratio as image_psnr

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

import tensorflow as tf
import keras
import keras.backend as K
from keras import Model
from keras import regularizers
from keras.layers import Input
from keras.layers import Dense, Conv1D, Conv1DTranspose
from keras.layers import LeakyReLU, PReLU
from keras.layers import Dropout, Flatten, Reshape, Concatenate, TimeDistributed
from keras.layers import MaxPooling1D, UpSampling1D, BatchNormalization, LayerNormalization
from keras.optimizers import Adam, Nadam
from tensorflow import expand_dims

my_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
K.clear_session()

def check_tf():
    sys_info, cuda_avail = tf.sysconfig.get_build_info(), tf.test.is_built_with_cuda()
    devices = tf.config.experimental.list_physical_devices('GPU')
    count, details = len(devices), tf.config.experimental.get_device_details(devices[0])
    cudnn_v = details['compute_capability']
    print('\n'+'-'*60)
    print('----------------------- VERSION INFO -----------------------')
    print('TensorFlow version: {} | TF Built with CUDA? {}'.format(tf.__version__, cuda_avail))
    print('# Device(s) available: {} | CUDA: {} | cuDNN: {}.{}'.format(count, sys_info['cuda_version'], cudnn_v[0], cudnn_v[1]))
    print('Name(s): {}'.format(details['device_name']))
    print('-'*60+'\n')
    return None

def check_torch():
    import torch
    torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
    cuda_v, cudnn_v = torch.version.cuda, torch.backends.cudnn.version()
    count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
    print('\n'+'-'*60)
    print('----------------------- VERSION INFO -----------------------')
    print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
    print('# Device(s) available: {} | CUDA: {} | cuDNN: {}.{} '.format(count,cuda_v,cudnn_v//1000,(cudnn_v%1000)/100))
    print('Name(s): {}'.format(name))
    print('-'*60+'\n')
    return None

def make_sparse_flowrates(flow, injection_idx):
    flowrates = np.zeros((200,4))
    flowrates[injection_idx[0]] = flow[0]
    flowrates[injection_idx[1]] = flow[1]
    flowrates[injection_idx[2]] = flow[2]
    flowrates[injection_idx[3]] = flow[3]
    flowrates[injection_idx[4]] = flow[4]
    return flowrates

def my_normalize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    data_norm = scaler.transform(data)
    return data_norm, scaler

def make_daslhs_dts(file_num, nsamples=None, range=(-1,1), save=False):
    # DTS
    dts = pd.read_pickle('data/dts_exp{}.pkl'.format(file_num))
    dts_norm = np.expand_dims(MinMaxScaler(range).fit_transform(dts.T),-1)
    # DAS
    das = pd.read_pickle('data/data_exp{}.pkl'.format(file_num))
    idx = np.sort(LatinHypercube(d=1).integers(l_bounds=0, u_bounds=das.shape[-1], n=dts.shape[1]).squeeze())
    das_lhs_norm = np.expand_dims(MinMaxScaler(range).fit_transform(das.iloc[:,idx].T),-1)
    if save:
        pd.to_pickle(das_lhs_norm, 'data/das{}_lhs.pkl'.format(file_num))
    return das_lhs_norm, dts_norm

def plot_loss(fit, figsize=None):
    epochs     = len(fit.history['loss'])
    iterations = np.arange(epochs)
    if figsize:
        plt.figure(figsize=figsize)
    plt.plot(iterations, fit.history['loss'],     '-', label='loss')
    plt.plot(iterations, fit.history['val_loss'], '-', label='validation loss')
    plt.title('Training: Loss vs epochs'); plt.legend()
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.xticks(iterations[::epochs//10])
    return None

def plot_relative_rates(data, cmap='Blues'):
    ylab = ['BG', 'Inj1', 'Inj2', 'Inj3', 'Inj4']
    xlab = ['oil', 'gas', 'water', 'sand']
    exp_titles = ['Exp 45', 'Exp 48', 'Exp 54', 'Exp 64', 'Exp 109', 'Exp 128']
    fig, axs = plt.subplots(1, 6, figsize=(15,3), facecolor='white')
    for i in range(6):
        im = axs[i].imshow(data[i], cmap=cmap)
        axs[i].set(title=exp_titles[i], 
                   xticks=range(4), xticklabels=xlab, 
                   yticks=range(5), yticklabels=ylab)
    for k in range(1,6):
        axs[k].set(yticks=[])
    cax = fig.add_axes([axs[-1].get_position().x1+0.01, axs[-1].get_position().y0,
                        0.02, axs[-1].get_position().y1-axs[-1].get_position().y0])
    plt.colorbar(im, cax=cax, label='relative rates')
    plt.show() 
    return None

def plot_relative_mat(data, cmap='Blues'):
    labels = ['Oil','Gas','Water','Sand']
    exp_titles = ['Exp 45', 'Exp 48', 'Exp 54', 'Exp 64', 'Exp 109', 'Exp 128']
    fig, axs = plt.subplots(1, 6, figsize=(15,3), facecolor='white')
    for i in range(6):
        im = axs[i].matshow(data[i], cmap=cmap, aspect=0.03)
        axs[i].set(title=exp_titles[i], xticks=np.arange(4), xticklabels=labels)
        axs[i].xaxis.set_ticks_position('bottom')
    for k in range(1,6):
        axs[k].set(yticks=[])
    axs[0].set(ylabel='Distance [m]')
    cax = fig.add_axes([axs[-1].get_position().x1+0.01, axs[-1].get_position().y0,
                        0.02, axs[-1].get_position().y1-axs[-1].get_position().y0])
    plt.colorbar(im, cax=cax, label='relative rates')
    plt.show()
    return None

def plot_das_dts_flow(das, dts, flow, figsize=(25,6), expnum=''):
    plt.figure(figsize=figsize, facecolor='white')
    plt.subplot(131)
    plt.imshow(das.squeeze().T, aspect='auto', cmap='seismic'); plt.colorbar()
    plt.title('Normalized DAS - Experiment {}'.format(expnum))
    plt.subplot(132)
    plt.imshow(dts.squeeze().T, aspect='auto', cmap='seismic'); plt.colorbar()
    plt.title('Normalized DTS - Experiment {}'.format(expnum))
    plt.subplot(133)
    plt.imshow(flow, cmap='gist_heat_r', aspect='auto'); plt.colorbar()
    plt.xticks([0,1,2,3], labels=['oil','water','gas','sand'])
    plt.title('Normalized Injection Rates & Points - Experiment {}'.format(expnum))
    plt.show()
    return None

def plot_latent(zdata, figsize=None, cmap='binary', vmin=0, vmax=1, title='Latent'):
    if figsize:
        plt.figure(figsize=figsize)
    plt.imshow(zdata.reshape((zdata.shape[0], zdata.shape[1]*zdata.shape[-1])).T, 
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title); plt.xlabel('pseudo-Timestep'); plt.ylabel('pseudo-Distance')
    plt.colorbar()
    return None

def plot_featuremaps(data, nrows, ncols, expnum='', cmap='afmhot', figsize=(20,6), vmin=None, vmax=None):
    k = 0
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].imshow(data[k], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            axs[i,j].set(xticks=[], yticks=[])
            k += 1
    plt.suptitle('Experiment {} Feature Maps'.format(expnum))
    plt.show()
    return None

def plot_true_pred_z(true, pred, latent, figsize=(30,6), filenum='', cmaps=['seismic','seismic'], vmin=-0.1, vmax=0.1):
    plt.figure(figsize=figsize, facecolor='white')
    plt.subplot(141)
    plt.imshow(true.squeeze().T, cmap=cmaps[0], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(); plt.title('Experiment {} True'.format(filenum))
    plt.subplot(142)
    plt.imshow(pred.squeeze().T, cmap=cmaps[0], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(); plt.title('Experiment {} Predicted'.format(filenum))
    plt.subplot(143)
    plt.imshow(np.abs(true.squeeze().T-pred.T), cmap=cmaps[1], aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(); plt.title('Experiment {} - Absolute Error'.format(filenum))
    plt.subplot(144)
    plot_latent(latent, title='Experiment {} Latent Space'.format(filenum))
    plt.show()
    return None

def plot_rates_true_pred(true, pred, expnum='', figsize=(20,5), vmin=0, vmax=1, cmaps=['turbo', 'gist_heat_r']):
    ticks, labels = [0,1,2,3], ['oil','water','gas','sand']
    plt.figure(figsize=figsize, facecolor='white')
    plt.subplot(131)
    plt.imshow(true, aspect='auto', cmap=cmaps[0])
    plt.title('Normalized Injection Rate Map - Exp {}'.format(expnum))
    plt.xticks(ticks, labels=labels); plt.ylabel('distance'); plt.colorbar()
    plt.subplot(132)
    plt.imshow(pred, aspect='auto', cmap=cmaps[0], vmin=vmin, vmax=vmax)
    plt.title('Predicted Injection Rate Map - Exp {}'.format(expnum))
    plt.xticks(ticks, labels=labels); plt.ylabel('distance'); plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.abs(true-pred), aspect='auto', cmap=cmaps[1], vmin=vmin, vmax=vmax)
    plt.title('Absolute Difference - Exp {}'.format(expnum))
    plt.xticks(ticks, labels=labels); plt.ylabel('distance'); plt.colorbar()
    plt.show()
    return None

###############################################################################################
def mse_ssim_loss(y_true, y_pred, alpha=0.8):
    mse  = tf.keras.losses.MeanSquaredError()
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2))
    return alpha*mse(y_true,y_pred) + (1-alpha)*ssim

def mse_mae_loss(y_true, y_pred, alpha=0.5):
    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()
    return alpha*mse(y_true,y_pred) + (1-alpha)*mae(y_true,y_pred)

def das_Unet(xsteps=200, act=LeakyReLU(alpha=0.3)):
    image = tf.keras.Input((xsteps,1), name='input')
    # downlayer 1
    conv1 = Conv1D(4, 3, activation=act, padding='same')(image)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(4, 3, activation=act, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    # downlayer 2
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(16, 3, activation=act, padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(16, 3, activation=act, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    # downlayer 3
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(32, 3, activation=act, padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(32, 3, activation=act, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    # downlayer 4
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    conv4 = Conv1D(64, 3, activation=act, padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(64, 3, activation=act, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    latent = conv4
    # uplayer 3
    up7 = Conv1DTranspose(32, 3, strides=2, padding='same')(conv4)
    up7 = Concatenate()([up7, conv3])
    conv7 = Conv1D(32, 3, activation=act, padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(32, 3, activation=act, padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    # uplayer 2
    up8 = Conv1DTranspose(16, 3, strides=2, padding='same')(conv7)
    up8 = Concatenate()([up8, conv2])
    conv8 = Conv1D(16, 3, activation=act, padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(16, 3, activation=act, padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    # uplayer 1
    up9 = Conv1DTranspose(4, 3, strides=2, padding='same')(conv8)
    up9 = Concatenate()([up9, conv1])
    conv9 = Conv1D(4, 3, activation=act, padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(4, 3, activation=act, padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    # outlayer
    out = Conv1D(1, 1, activation='linear')(conv9)
    # models
    das_m2m = Model(inputs=[image], outputs=[out])
    das_m2z = Model(inputs=[image], outputs=[latent])
    return das_m2m, das_m2z

def dts_Unet(xsteps=200, act=LeakyReLU(alpha=0.3)):
    image = tf.keras.Input((xsteps,1), name='input')
    # downlayer 1
    conv1 = Conv1D(4, 3, activation=act, padding='same')(image)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(4, 3, activation=act, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    # downlayer 2
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(16, 3, activation=act, padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(16, 3, activation=act, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    # downlayer 3
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(32, 3, activation=act, padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(32, 3, activation=act, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    # downlayer 4
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    conv4 = Conv1D(64, 3, activation=act, padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(64, 3, activation=act, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    latent = conv4
    # uplayer 3
    up7 = Conv1DTranspose(32, 3, strides=2, padding='same')(conv4)
    up7 = Concatenate()([up7, conv3])
    conv7 = Conv1D(32, 3, activation=act, padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(32, 3, activation=act, padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    # uplayer 2
    up8 = Conv1DTranspose(16, 3, strides=2, padding='same')(conv7)
    up8 = Concatenate()([up8, conv2])
    conv8 = Conv1D(16, 3, activation=act, padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(16, 3, activation=act, padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    # uplayer 1
    up9 = Conv1DTranspose(4, 3, strides=2, padding='same')(conv8)
    up9 = Concatenate()([up9, conv1])
    conv9 = Conv1D(4, 3, activation=act, padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(4, 3, activation=act, padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    # outlayer
    out = Conv1D(1, 1, activation='linear')(conv9)
    # models
    dts_m2m = Model(image, out)
    dts_m2z = Model(image, latent)
    return dts_m2m, dts_m2z

def make_flowpred_from_dual_latent(zdas, zdts, flow, xsteps=200, expnum='', 
                                   method=LinearRegression(), ssim_window=3, 
                                   plot=True, figsize=(10,4), cmaps=['gist_heat_r','hot_r']):
    z_dual = np.concatenate([zdas, zdts]).flatten().reshape(xsteps,-1)
    reg = method
    reg.fit(z_dual, flow)
    flow_pred_f = reg.predict(z_dual)
    flow_pred   = np.reshape(flow_pred_f, flow.shape)
    print('MSE:  {:.2e}'.format(mean_squared_error(flow, flow_pred_f)))
    print('SSIM: {:.3f}'.format(image_ssim(flow, flow_pred, win_size=ssim_window, data_range=1.0)))
    if plot:
        titles = ['True Relative Rates - Exp {}'.format(expnum), 
                  'Predicted Relative Rates - Exp {}'.format(expnum),
                  'Percent Error - Exp {}'.format(expnum)]
        xlabels = ['Oil','Gas','Water','Sand']
        err = np.abs(np.divide((flow_pred-flow), np.where(flow==0.,np.nan,flow)))*100
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        im0 = axs[0].imshow(flow, aspect='auto', cmap=cmaps[0], vmin=0, vmax=1)
        im1 = axs[1].imshow(flow_pred, aspect='auto', cmap=cmaps[0], vmin=0, vmax=1)
        im2 = axs[2].imshow(err, aspect='auto', cmap=cmaps[1])
        plt.colorbar(im0, ax=axs[0]); plt.colorbar(im1, ax=axs[1]); plt.colorbar(im2, ax=axs[2])
        for i in range(3):
            axs[i].set(title=titles[i], xticks=np.arange(4), xticklabels=xlabels, ylim=(xsteps,0))
            axs[i].vlines([np.arange(4)+0.5], 0, xsteps, color='k', ls='--', alpha=0.2)
        plt.tight_layout()
        plt.show()
    return reg

def transfer_learning_predictions_dual(newdas, newdts, newflow, dasm2z, dtsm2z, expnum,
                                  xsteps=200, method=LinearRegression(), ssim_window=3, 
                                  plot=True, figsize=(10,4), cmaps=['gist_heat_r','hot_r']):
    newdas_z = dasm2z.predict(newdas, verbose=0).squeeze().astype('float64')
    newdts_z = dtsm2z.predict(newdts, verbose=0).squeeze().astype('float64')
    print('Shapes - z_DAS: {} | z_DTS: {}'.format(newdas_z.shape, newdts_z.shape))
    make_flowpred_from_dual_latent(newdas_z, newdts_z, newflow, 
                                   xsteps=xsteps, expnum=expnum, method=method, 
                                   ssim_window=ssim_window, plot=plot, 
                                   figsize=figsize, cmaps=cmaps)
    return None

############### Single Latent Spaces ###############
def make_single_latents(models, data, keys=['45','48','54','64','109','128']):
    das1, das2, das3, das4, das5, das6 = data['das'].values()
    dts1, dts2, dts3, dts4, dts5, dts6 = data['dts'].values()

    das_m2m, das_m2z = models['das']['m2m'], models['das']['m2z']
    dts_m2m, dts_m2z = models['dts']['m2m'], models['dts']['m2z']

    das1_z = das_m2z.predict(das1, verbose=0).squeeze().astype('float64')
    das2_z = das_m2z.predict(das2, verbose=0).squeeze().astype('float64')
    das3_z = das_m2z.predict(das3, verbose=0).squeeze().astype('float64')
    das4_z = das_m2z.predict(das4, verbose=0).squeeze().astype('float64')
    das5_z = das_m2z.predict(das5, verbose=0).squeeze().astype('float64')
    das6_z = das_m2z.predict(das6, verbose=0).squeeze().astype('float64')
    print('DAS Latent Spaces: \n'+'-'*57)
    print('{}: {} | {}: {}   | {}: {}'.format(keys[0], das1_z.shape, keys[1], das2_z.shape, keys[2], das3_z.shape))
    print('{}: {} | {}: {} | {}: {}'.format(keys[3], das4_z.shape, keys[4], das5_z.shape, keys[5], das6_z.shape))
    print('-'*57)

    dts1_z = dts_m2z.predict(dts1, verbose=0).squeeze().astype('float64')
    dts2_z = dts_m2z.predict(dts2, verbose=0).squeeze().astype('float64')
    dts3_z = dts_m2z.predict(dts3, verbose=0).squeeze().astype('float64')
    dts4_z = dts_m2z.predict(dts4, verbose=0).squeeze().astype('float64')
    dts5_z = dts_m2z.predict(dts5, verbose=0).squeeze().astype('float64')
    dts6_z = dts_m2z.predict(dts6, verbose=0).squeeze().astype('float64')
    print('\nDTS Latent Spaces: \n'+'-'*57)
    print('{}: {} | {}: {}   | {}: {}'.format(keys[0], dts1_z.shape, keys[1], dts2_z.shape, keys[2], dts3_z.shape))
    print('{}: {} | {}: {} | {}: {}'.format(keys[3], dts4_z.shape, keys[4], dts5_z.shape, keys[5], dts6_z.shape))
    print('-'*57)

    dasz = {'45':das45_z, '48':das48_z, '54':das54_z, '64':das64_z, '109':das109_z, '128':das128_z}
    dtsz = {'45':dts45_z, '48':dts48_z, '54':dts54_z, '64':dts64_z, '109':dts109_z, '128':dts128_z}
    return {'das':dasz, 'dts':dtsz}

def make_flowpred_from_single_latent(latents:dict, flow:dict, expnum:str='', xsteps=200,
                                     method=LinearRegression(), ssim_window=3,
                                     plot=True, figsize=(12, 7), cmaps=['gist_heat_r','binary']):
    flow = flow[expnum]
    zdas = latents['das'][expnum].flatten().reshape(xsteps,-1)
    regdas = method
    regdas.fit(zdas,flow)
    flow_pred_f_das = regdas.predict(zdas)
    flow_pred_das   = np.reshape(flow_pred_f_das, flow.shape)
    flow_err_das = np.abs(flow-flow_pred_das)
    print('DAS only: MSE={:.2e}, SSIM={:.3f}'.format(mean_squared_error(flow, flow_pred_f_das),
                                                     image_ssim(flow, flow_pred_das, win_size=ssim_window, data_range=1.0)))

    zdts = latents['dts'][expnum].flatten().reshape(xsteps,-1)
    regdts = method
    regdts.fit(zdts,flow)
    flow_pred_f_dts = regdts.predict(zdts)
    flow_pred_dts   = np.reshape(flow_pred_f_dts, flow.shape)
    flow_err_dts = np.abs(flow-flow_pred_dts)
    print('DTS only: MSE={:.2e}, SSIM={:.3f}'.format(mean_squared_error(flow, flow_pred_f_dts),
                                                     image_ssim(flow, flow_pred_dts, win_size=ssim_window, data_range=1.0)))

    zdual = np.concatenate([zdas, zdts]).flatten().reshape(xsteps,-1)
    regdual = method
    regdual.fit(zdual,flow)
    flow_pred_f = regdual.predict(zdual)
    flow_pred   = np.reshape(flow_pred_f, flow.shape)
    flow_err = np.abs(flow-flow_pred)
    print('Dual:     MSE={:.2e}, SSIM={:.3f}'.format(mean_squared_error(flow, flow_pred_f),
                                                     image_ssim(flow, flow_pred, win_size=ssim_window, data_range=1.0)))

    if plot:
        xlabels = ['Oil','Gas','Water','Sand']
        pred = [flow_pred_das, flow_pred_dts, flow_pred]
        err  = [flow_err_das, flow_err_dts, flow_err]
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 5, figure=fig, width_ratios=[0.75, 1, 1, 1, 0.1])
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2]); ax4 = fig.add_subplot(gs[0, 3])
        ax5 = fig.add_subplot(gs[1, 1]); ax6 = fig.add_subplot(gs[1, 2]); ax7 = fig.add_subplot(gs[1, 3])
        cax1 = fig.add_subplot(gs[:1, 4])
        cax2 = fig.add_subplot(gs[1:, 4])
        axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        ax1.imshow(flow, cmap=cmap[0], aspect='auto', interpolation='none')
        for k, ax in enumerate([ax2, ax3, ax4]):
            im1 = ax.imshow(pred[k], cmap=cmaps[0], aspect='auto', interpolation='none', vmin=0, vmax=1)
        cb1 = plt.colorbar(im1, cax=cax1)
        for k, ax in enumerate([ax5, ax6, ax7]):
            im2 = ax.imshow(err[k], cmap=cmaps[1], aspect='auto', interpolation='none', vmin=0, vmax=5e-3)
        cb2 = plt.colorbar(im2, cax=cax2)
        for ax in [ax2, ax3, ax4]:
            ax.set_xticks([])
        for ax in [ax3, ax4, ax6, ax7]:
            ax.set_yticks([])
        for ax in axs:
            ax.set_ylim(0,xsteps)
            ax.invert_yaxis()
            ax.vlines(range(4), 0, xsteps, color='k', ls='--', alpha=0.25)
        for ax in [ax1, ax5, ax6, ax7]:
            ax.set_xticks(range(4))
            ax.set_xticklabels(xlabels, weight='bold')
        for ax in [ax1, ax2, ax5]:
            ax.set_ylabel('Distance [m]')
        ax2.set_title('DAS only', weight='bold')
        ax3.set_title('DTS only', weight='bold')
        ax4.set_title('Dual', weight='bold')
        ax1.set_title('True Relative Rates', weight='bold')
        ax41 = ax4.twinx(); ax41.set_yticks([]); ax41.set_ylabel('Predicted Relative Rates', weight='bold', labelpad=20, rotation=270, fontsize=12)
        ax71 = ax7.twinx(); ax71.set_yticks([]); ax71.set_ylabel('Absolute Error', weight='bold', labelpad=20, rotation=270, fontsize=12)
        plt.suptitle('Experiment {}'.format(expnum), fontsize=16, weight='bold')
        plt.tight_layout(); plt.show()
    return None

def make_uq_pred_dual(expnum:str, all_data:dict, models:dict, flow_dict:dict, noise_lvl:list=[5, 10, 25, 50], 
                      method = LinearRegression(), ssim_window=3,
                      plot:bool=True, figsize=(15,7.5),
                      cmap='gist_heat_r', cmap2='binary'):
    flow  = flow_dict[expnum]
    das   = all_data['das'][expnum]
    dts   = all_data['dts'][expnum]
    noise = np.random.normal(0, 1, das.shape)
    if plot:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 5, height_ratios=[1, 1, .0001])
        xlabels = ['Oil','Gas','Water','Sand']
        ax0 = fig.add_subplot(gs[:-1, 0])
        im0 = ax0.imshow(flow, aspect='auto', cmap=cmap)
        ax0.set(xticks=np.arange(4), xticklabels=xlabels, ylabel='Distance [m]')
        ax0.set_title('Trial {}'.format(expnum), weight='bold')
        plt.colorbar(im0)
        ax11 = fig.add_subplot(gs[0, 1])
        ax12 = fig.add_subplot(gs[0, 2])
        ax13 = fig.add_subplot(gs[0, 3])
        ax14 = fig.add_subplot(gs[0, 4])
        top_axs = [ax11, ax12, ax13, ax14]
        ax21 = fig.add_subplot(gs[1, 1])
        ax22 = fig.add_subplot(gs[1, 2])
        ax23 = fig.add_subplot(gs[1, 3])
        ax24 = fig.add_subplot(gs[1, 4])
        bot_axs = [ax21, ax22, ax23, ax24]
        ax31 = fig.add_subplot(gs[2, 1])
        ax32 = fig.add_subplot(gs[2, 2])
        ax33 = fig.add_subplot(gs[2, 3])
        ax34 = fig.add_subplot(gs[2, 4])
        txt_axs = [ax31, ax32, ax33, ax34]
        for i in range(4):
            das_n = das + noise * noise_lvl[i]*das.std()
            dts_n = dts + noise * noise_lvl[i]*dts.std()
            z_das = models['das']['m2z'].predict(das_n, verbose=0).squeeze().astype('float64')
            z_dts = models['dts']['m2z'].predict(dts_n, verbose=0).squeeze().astype('float64')
            z_dual = np.concatenate([z_das, z_dts]).flatten().reshape(200,-1)
            reg = method
            reg.fit(z_dual, flow)
            flow_pred_f = reg.predict(z_dual)
            flow_pred   = np.reshape(flow_pred_f, flow.shape)
            err = np.abs(flow-flow_pred)
            mse = image_mse(flow, flow_pred)
            ssim = image_ssim(flow, flow_pred, win_size=ssim_window, data_range=1.0)
            im1 = top_axs[i].imshow(flow_pred, aspect='auto', cmap=cmap, vmin=0, vmax=1)
            top_axs[i].set(xticks=np.arange(4), xticklabels=xlabels, title='Prediction - {:.0f}% Noise'.format(noise_lvl[i]))
            im2 = bot_axs[i].imshow(err, aspect='auto', cmap=cmap2, vmin=0, vmax=6e-15)
            bot_axs[i].set(xticks=np.arange(4), xticklabels=xlabels, title='Absolute Error')
            plt.colorbar(im1); plt.colorbar(im2)
            #im3 = txt_axs[i].text(0.5, 0.5, 'MSE:  {:.2e}\nSSIM: {:.3f}'.format(mse, ssim), ha='center', va='center', transform=txt_axs[i].transAxes)
            txt_axs[i].axis('off')
        plt.tight_layout(); plt.show()
    return None


### OLD FUNCTIONS ###
# Open raw DAS/DTS files and save as pandas pickles
# crop fiber data into 200m segment, corresponding to the length of the flow-loop [4950, 5150]

# def read_save_dts_data(folder, save_name, size=200):
#     print('This is Experiment {}. Saved.'.format(folder[-2:]))
#     ### folder1 == 'E:/Research/Lytt Fiber Optics/DTS Experiment 54' ###
#     ### folder2 == 'E:/Research/Lytt Fiber Optics/DTS Experiment 64' ###
#     # save depth datums
#     dts_depths = pd.read_csv(os.path.join(folder, os.listdir(folder)[0]), usecols=[1]).squeeze()[-size:]
#     dts_depths.to_pickle('dts_depthstamps.pkl')
#     # save timestamps
#     dts_timestamps = pd.Series(dtype='object')
#     for i in range(len(os.listdir(folder))):
#         dts_timestamps.loc[i] = pd.to_datetime(os.listdir(folder)[i][9:19], format="%H%M%S%f").time()
#     if folder[-2:]=='54':
#         dts_timestamps.to_pickle('dts_exp54_timestamps.pkl')    
#     elif folder[-2:]=='64':
#         dts_timestamps.to_pickle('dts_exp64_timestamps.pkl')
#     # save temperature data
#     all_files  = os.listdir(folder)
#     dts_df = pd.DataFrame(())
#     for i in range(len(all_files)):
#         my_file = os.listdir(folder)[i]
#         file_path = os.path.join(folder, my_file)
#         new_dts   = pd.read_csv(file_path, usecols=[2])
#         dts_df = pd.concat([dts_df, new_dts], ignore_index=True, axis=1)
#     dts_postprocess = dts_df.iloc[-size:]
#     dts_postprocess.to_pickle(save_name)

# # Function to open H5 file as Pandas DataFrame
# def open_fiber_H5_2_arr(folder_n=1, file_n=0, xstart=4950, xend=5150):
#     if folder_n==1:
#         fold = 'E:/Lytt Fiber Optics/Sintef3mH5'
#     elif folder_n==2:
#         fold = 'E:/Lytt Fiber Optics/Sintef10mH5'
#     elif folder_n==45:
#         fold = 'E:/Lytt Fiber Optics/45'
#     elif folder_n==48:
#         fold = 'E:/Lytt Fiber Optics/48'
#     elif folder_n==109:
#         fold = 'E:/Lytt Fiber Optics/109'
#     elif folder_n==128:
#         fold = 'E:/Lytt Fiber Optics/128'
#     file = os.listdir(fold)
#     file_path = os.path.join(fold, file[file_n])
#     f = h5py.File(file_path, 'r')
#     df = pd.DataFrame((f['Acquisition']['Raw[0]']['RawData'])).iloc[:, xstart:xend].T
#     f.close
#     return df

# # change folder_i, files_i to desired experiment
# folder128 = 'E:/Lytt Fiber Optics/128'
# print('Experiment 128 (file128) # of files: {}'.format(len(os.listdir(folder128))))
# data_exp128 = pd.DataFrame(())
# for i in range(len(os.listdir(folder128))):
#     new_data1 = open_fiber_H5_2_arr(folder_n=128, file_n=i, xstart=4950, xend=5150)
#     data_exp128 = pd.concat([data_exp128, new_data1], ignore_index=True, axis=1)
#     data_exp128.to_pickle('E:/Lytt Fiber Optics/data_exp128.pkl')