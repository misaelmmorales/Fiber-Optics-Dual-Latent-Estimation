import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import h5py                                           #import h5 files
import os                                             #OS operations
import time                                           #timing and clock time
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

import keras
import keras.backend as K
from keras import Model
from keras import regularizers
from keras.layers import Input
from keras.layers import LeakyReLU, PReLU
from keras.layers import Dropout, Flatten, Reshape, Concatenate, TimeDistributed, Concatenate
from keras.layers import Dense
from keras.layers import Conv1D, Conv1DTranspose
from keras.layers import MaxPooling1D, UpSampling1D, BatchNormalization
from keras.optimizers import Adam, Nadam
from tensorflow_addons.layers import InstanceNormalization
from keras.layers import LayerNormalization
from tensorflow import expand_dims

# Define arguments for text box in PLT.TEXT()
my_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

def check_tf():
    sys_info = tf.sysconfig.get_build_info()
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())

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
    dts = pd.read_pickle('dts_exp{}.pkl'.format(file_num))
    dts_norm = np.expand_dims(MinMaxScaler(range).fit_transform(dts.T),-1)
    # DAS
    das = pd.read_pickle('data_exp{}.pkl'.format(file_num))
    idx = np.sort(LatinHypercube(d=1).integers(l_bounds=0, u_bounds=das.shape[-1], n=dts.shape[1]).squeeze())
    das_lhs_norm = np.expand_dims(MinMaxScaler(range).fit_transform(das.iloc[:,idx].T),-1)
    if save:
        pd.to_pickle(das_lhs_norm, 'das{}_lhs.pkl'.format(file_num))
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

def plot_relative_mat(data, cmap='Blues'):
    labels = ['Oil','Gas','Water','Sand']
    exp_titles = ['Exp 45', 'Exp 48', 'Exp 54', 'Exp 64', 'Exp 109', 'Exp 128']
    fig, axs = plt.subplots(1, 6, figsize=(15,3), facecolor='white')
    for i in range(6):
        im = axs[i].matshow(data[i], cmap=cmap, aspect=0.03)
        axs[i].set(title=exp_titles[i], xticks=np.arange(4), xticklabels=labels)
    for k in range(1,6):
        axs[k].set(yticks=[])
    axs[0].set(ylabel='Distance [m]')
    cax = fig.add_axes([axs[-1].get_position().x1+0.01, axs[-1].get_position().y0,
                        0.02, axs[-1].get_position().y1-axs[-1].get_position().y0])
    plt.colorbar(im, cax=cax, label='relative rates')
    plt.show()

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

def plot_latent(zdata, figsize=None, cmap='binary', vmin=0, vmax=1, title='Latent'):
    if figsize:
        plt.figure(figsize=figsize)
    plt.imshow(zdata.reshape((zdata.shape[0], zdata.shape[1]*zdata.shape[-1])).T, 
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title); plt.xlabel('pseudo-Timestep'); plt.ylabel('pseudo-Distance')
    plt.colorbar()

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

###############################################################################################
def mse_ssim_loss(y_true, y_pred, alpha=0.8):
    mse  = tf.keras.losses.MeanSquaredError()
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2))
    return alpha*mse(y_true,y_pred) + (1-alpha)*ssim

def mse_mae_loss(y_true, y_pred, alpha=0.5):
    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()
    return alpha*mse(y_true,y_pred) + (1-alpha)*mae(y_true,y_pred)

def das_Unet(act=LeakyReLU(alpha=0.3)):
    image = tf.keras.Input((200,1), name='input')
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

def dts_Unet(act=LeakyReLU(alpha=0.3)):
    image = tf.keras.Input((200,1), name='input')
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

def make_flowpred_from_dual_latent(zdas, zdts, flow, expnum='', 
                                   method=LinearRegression(), ssim_window=3, 
                                   plot=True, figsize=(10,4), cmap='gist_heat_r'):
    z_dual = np.concatenate([zdas, zdts]).flatten().reshape(200,-1)
    reg = method
    reg.fit(z_dual, flow)
    flow_pred_f = reg.predict(z_dual)
    flow_pred   = np.reshape(flow_pred_f, flow.shape)
    print('MSE:  {:.2e}'.format(mean_squared_error(flow, flow_pred_f)))
    print('SSIM: {:.3f}'.format(image_ssim(flow, flow_pred, win_size=ssim_window)))
    if plot:
        titles = ['True Relative Rates - Exp {}'.format(expnum), 
                  'Predicted Relative Rates - Exp {}'.format(expnum)]
        xlabels = ['Oil','Gas','Water','Sand']
        plt.figure(figsize=figsize)
        k = 0
        for i in [flow, flow_pred]:
            plt.subplot(1,2,k+1)
            plt.imshow(i, aspect='auto', cmap=cmap)
            plt.colorbar(label='relative rates'); plt.clim(0,1)
            plt.title(titles[k]); plt.xticks(np.arange(4), labels=xlabels)
            k += 1
    return reg

def transfer_learning_predictions(newdas, newdts, newflow, dasm2z, dtsm2z, expnum, 
                                  method=LinearRegression(), ssim_window=3, 
                                  plot=True, figsize=(10,4), cmap='gist_heat_r'):
    newdas_z = dasm2z.predict(newdas).squeeze().astype('float64')
    newdts_z = dtsm2z.predict(newdts).squeeze().astype('float64')
    print('Shapes - z_DAS: {} | z_DTS: {}'.format(newdas_z.shape, newdts_z.shape))
    make_flowpred_from_dual_latent(newdas_z, newdts_z, newflow, 
                                   expnum=expnum, method=method, 
                                   ssim_window=ssim_window, plot=plot, 
                                   figsize=figsize, cmap=cmap)