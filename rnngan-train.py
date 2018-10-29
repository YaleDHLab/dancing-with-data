#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
import pickle

import keras.backend as K
from keras import layers
from keras.models import Model
from keras.utils import to_categorical
import tensorflow as tf

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

def load_data(file_name):
    X0 = np.load(file_name)

    input_file = os.path.basename(file_name)
    if input_file == 'mariel-1.npy':
        X0 = X0.transpose((1,0,2))
        X0 = X0[1300:7000]
        X0[:,:,2] *= -1
        X = X0.copy()
        X -= X.mean(axis=(0,1))
        X *= 0.5
    elif input_file == 'carrie-10-min-scaled.npy':
        X0 = X0.transpose((1,0,2))
        X = X0.copy()
        X[:,:,1] = X0[:,:,2]
        X[:,:,2] = X0[:,:,1]
        X = X[100:]
        X[:,:,:2] -= 0.5
    if input_file == 'carrie-10-mins.npy':
        X0 = X0.transpose((1,0,2))
        X0 = X0[100:]
        X0[:,:,2] *= -1
        X = X0.copy()
        X -= X.mean(axis=(0,1))
        X *= 0.25

    return X

def mk_model(X, lookback=60, npred=4, n_cells=(48,48,48), n_cells_disc=(48,48,), noise_dim=128):
    K.clear_session()
    
    input_seq = layers.Input((lookback, X.shape[1], X.shape[2]))
    H = input_seq
    
    input_noise = layers.Input((noise_dim,))
    
    H = layers.Reshape((lookback, X.shape[1]*X.shape[2]))(H)
    for nc in n_cells[:-1]:
        H = layers.CuDNNLSTM(nc, return_sequences=True)(H)
    H = layers.CuDNNLSTM(n_cells[-1])(H)
    H = layers.Concatenate()([H, input_noise])
    H = layers.Dense(npred*X.shape[1]*X.shape[2], activation='tanh')(H)
    H = layers.Reshape((npred,X.shape[1],X.shape[2]))(H)
    output_seq = H
    
    generator = Model([input_seq, input_noise], output_seq)
    
    disc_input1 = layers.Input((lookback, X.shape[1], X.shape[2]))
    disc_input2 = layers.Input((npred, X.shape[1], X.shape[2]))
    
    H1 = layers.Reshape((lookback, X.shape[1]*X.shape[2]))(disc_input1)
    H2 = layers.Reshape((npred, X.shape[1]*X.shape[2]))(disc_input2)
    H = layers.Concatenate(axis=1)([H1,H2])
    
    for nc in n_cells_disc[:-1]:
        H = layers.CuDNNLSTM(nc, return_sequences=True)(H)
    H = layers.CuDNNLSTM(n_cells_disc[-1])(H)

    #H = layers.Dense(32, activation='sigmoid')(H)

    HH = layers.Dense(32, activation='sigmoid')(H2)
    HH = layers.Dense(32, activation='sigmoid')(H2)
    HH = layers.Flatten()(HH)

    H = layers.Concatenate()([H,HH])
    H = layers.Dense(2, activation='softmax')(H)
    disc_output = H
    
    discriminator = Model([disc_input1,disc_input2], disc_output)
    discriminator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    discriminator.trainable = False
    full_output = discriminator([input_seq,generator([input_seq, input_noise])])
    full = Model([input_seq, input_noise], full_output)
    full.compile(loss='categorical_crossentropy', optimizer='adam')
    discriminator.trainable = True
    
    return generator, discriminator, full

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotate", action='store_true', help="Apply rotations during training")
    parser.add_argument("--lr-gen", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--lr-disc", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lookback", type=int, default=24, help="Number of lookback frames")
    parser.add_argument("--npred", type=int, default=4, help="Number of forward frames to predict")
    parser.add_argument("--noise-dim", type=int, default=128, help="Noise dimensionality")
    parser.add_argument("--gpu", type=str, help="GPU selector")
    parser.add_argument("--data", default="carrie-10-mins.npy", help="Data to train on")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--force", action='store_true', help="Force overwrite directories")
    parser.add_argument("path", help="Output path")
    args = parser.parse_args()

    # config for Gatsby cores
    if 'gatsby' in os.environ['HOSTNAME']:
        if not 'CUDA_VISIBLE_DEVICES' in os.environ:
            # specify target gpu device
            os.environ['CUDA_VISIBLE_DEVICES'] = '0' # either '0' or '1' to utilize Titan X GPUs

        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        # allow dynamic GPU allocation
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

    dirname = 'rnngan_rot%d_lrg%g_lrd%g_b%d'%(args.rotate, args.lr_gen,args.lr_disc,args.batch_size)
    output_path = os.path.join(args.path, dirname)

    try:
        os.makedirs(output_path)
    except OSError as e:
        import errno
        if e.errno == errno.EEXIST:
            if args.force:
                print("Warning -- writing results to existing path:", output_path)
            else:
                print("Exiting because of existing path:", output_path)
                sys.exit()
        else:
            raise

    with open(os.path.join(output_path, 'config.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    with open(os.path.join(output_path, 'pid.%d'%os.getpid()), 'w') as f:
        pass

    print("Loading data")
    X = load_data(args.data)

    epochs = args.epochs
    batch_size = args.batch_size
    do_rotation = args.rotate
    lookback = args.lookback
    npred = args.npred
    noise_dim = args.noise_dim

    gen, disc, full = mk_model(X, lookback=lookback, npred=npred, noise_dim=noise_dim)

    K.set_value(full.optimizer.lr, args.lr_gen)
    K.set_value(disc.optimizer.lr, args.lr_disc)

    d_losses = []
    g_losses = []

    nbatches = (len(X)-batch_size*lookback-npred)//batch_size

    y_real = to_categorical(np.ones(batch_size),2)
    y_fake = to_categorical(np.zeros(batch_size),2)
    print("Nbatches =", nbatches)
    for iepoch in range(epochs):
        offsets = np.random.choice(len(X)-batch_size*lookback-npred, replace=False, size=(nbatches,batch_size))
        
        d_loss = 0
        g_loss = 0
        for ibatch in range(nbatches):
            batch_idxs = offsets[ibatch].repeat(lookback).reshape(batch_size,lookback) + np.arange(lookback)
            truth_idxs = offsets[ibatch].repeat(npred).reshape(batch_size,npred) + np.arange(npred) + lookback
            
            if do_rotation:
                theta = 2*np.pi*np.random.rand()
                c,s = np.cos(theta), np.sin(theta)
                rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                x1_real = rot.dot(X[batch_idxs].transpose((0,1,3,2))).transpose((1,2,3,0))
                x2_real = rot.dot(X[truth_idxs].transpose((0,1,3,2))).transpose((1,2,3,0))
            else:
                x1_real = X[batch_idxs]
                x2_real = X[truth_idxs]
            
            noise = np.random.normal(size=(batch_size,noise_dim))
            x2_fake = gen.predict([x1_real, noise])
            
            disc.trainable = True
            l, a = disc.train_on_batch([x1_real, x2_real], y_real)
            d_loss += 0.5*l
            l,a = disc.train_on_batch([x1_real, x2_fake], y_fake)
            d_loss += 0.5*l
            
            if do_rotation:
                theta = 2*np.pi*np.random.rand()
                c,s = np.cos(theta), np.sin(theta)
                rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                x1_real = rot.dot(X[batch_idxs].transpose((0,1,3,2))).transpose((1,2,3,0))
            
            disc.trainable = False
            noise = np.random.normal(size=(batch_size,noise_dim))
            g_loss += full.train_on_batch([x1_real, noise], y_real)
            
        d_losses.append(d_loss/nbatches)
        g_losses.append(g_loss/nbatches)
        print("Epoch %d/%d: L(d)=%.2e L(g)=%.2e" % (iepoch, epochs, d_losses[-1], g_losses[-1]))
        sys.stdout.flush()

        if iepoch%10==0:
            fig = plt.figure(0)
            plt.clf()
            plt.plot(d_losses, label='disc')
            plt.plot(g_losses, label='gen')
            plt.legend()
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('loss', fontsize=14)
            fname = os.path.join(output_path, 'losses.png')
            plt.savefig(fname)
            print("Saved losses to", fname)

            fig = plt.figure(1)
            plt.clf()
            examples = gen.predict([x1_real[:4], np.random.normal(size=(4,args.noise_dim))])
            for iplt in range(4):
                ax = fig.add_subplot(2,2,iplt+1,projection='3d')
                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
                ax.set_zlim(-1,1)
                ax.scatter(examples[iplt,0,:,0], examples[iplt,0,:,1], examples[iplt,0,:,2])
            plt.suptitle("Epoch %d"%iepoch)
            fname = os.path.join(output_path, 'samples_e%03d.png'%iepoch)
            plt.savefig(fname)
            print("Saved samples to", fname)
            sys.stdout.flush()

        if iepoch%10==0 or iepoch==(args.epochs-1):
            model_file = os.path.join(output_path, 'latest_model.h5')
            weights_file = os.path.join(output_path, 'latest_weights.h5')
            full.save(model_file)
            full.save_weights(weights_file)
            print("Saved model to", model_file)
            print("Saved weights to", weights_file)
            sys.stdout.flush()
