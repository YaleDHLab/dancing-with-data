#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
import pickle
import itertools as it

import keras.backend as K
from keras import layers
from keras.models import Model
from keras.utils import to_categorical
import tensorflow as tf

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

from utils.plot import plot_pose, plot_stick

def load_data(file_name):
    X0 = np.load(file_name)

    # load up the input data and do some basic cleaning operations
    input_file = os.path.basename(file_name)
    if input_file == 'mariel-1.npy':
        X0 = X0.transpose((1,0,2))
        X0 = X0[1300:7000]
        X0[:,:,2] *= -1
        X = X0.copy()
        X -= X.mean(axis=(0,1))
        X *= 0.5
    elif input_file == 'carrie-10-mins.npy':
        X0 = X0.transpose((1,0,2))
        X0 = X0[100:]
        X0[:,:,2] *= -1
        X = X0.copy()
        X -= X.mean(axis=(0,1))
        X *= 0.5


    # calculate pairwise distance stats for rigid
    # edge detection
    vdist_mean = np.zeros((X.shape[1],X.shape[1]))
    vdist_var = np.zeros_like(vdist_mean)
    for i,j in it.combinations(range(X.shape[1]), r=2):
        vdist = np.sum((X[:,i]-X[:,j])**2, axis=-1)
        vdist_mean[i,j] = vdist_mean[j,i] = vdist.mean()
        vdist_var[i,j] = vdist_var[j,i] = vdist.var(ddof=1)

    # sort vertex pairs by lowest variance
    upper_triangle = np.triu_indices_from(vdist_var, k=1)
    variances = vdist_var[upper_triangle]
    vtx_pairs = [(i,j,vdist_mean[i,j],vdist_var[i,j]) for i,j in zip(*upper_triangle)]
    vtx_pairs.sort(key=lambda v: v[3])

    return X, vtx_pairs

def mk_model(X, n_prompt=60, n_pred=4, n_cells=(96,64,32), n_cells_disc=(48,48,48),
             noise_dim=32, center_seq=True, rigid_edges=None):
    K.clear_session()
    
    input_seq = layers.Input((n_prompt, X.shape[1], X.shape[2]))
    H = input_seq
    
    input_noise = layers.Input((noise_dim,))
    Hn = input_noise

    if center_seq:
        offsets = layers.Lambda(lambda x: K.constant([[[[1,1,0]]]])*K.mean(x, axis=(1,2), keepdims=True))(input_seq)
        H = layers.Subtract()([H, offsets])

    H = layers.Reshape((n_prompt, X.shape[1]*X.shape[2]))(H)

    for nc in n_cells:
        H = layers.CuDNNLSTM(nc, return_sequences=True)(H)
    H = layers.Cropping1D((n_prompt-n_pred,0))(H)

    Hn = layers.Dense(n_pred*n_cells[-1])(Hn)
    Hn = layers.PReLU()(Hn)
    Hn = layers.Reshape((n_pred, n_cells[-1]))(Hn)

    H = layers.Add()([H, Hn])

    H = layers.Dense(X.shape[1]*X.shape[2], activation='tanh')(H)

    H = layers.Reshape((n_pred,X.shape[1],X.shape[2]))(H)

    if center_seq:
        H = layers.Add()([H, offsets])

    generator_output = H

    generator = Model([input_seq, input_noise], generator_output, name='generator')


    disc_input1 = H1 = layers.Input((n_prompt, X.shape[1], X.shape[2]))
    disc_input2 = H2 = layers.Input((n_pred, X.shape[1], X.shape[2]))

    if center_seq:
        doffsets = layers.Lambda(lambda x: K.constant([[[[1,1,0]]]])*K.mean(x, axis=(1,2), keepdims=True))(disc_input1)
        H1 = layers.Subtract()([H1, doffsets])
        H2 = layers.Subtract()([H2, doffsets])

    H1 = layers.Reshape((n_prompt, X.shape[1]*X.shape[2]))(H1)
    H2 = layers.Reshape((n_pred, X.shape[1]*X.shape[2]))(H2)

    for nc in n_cells_disc[:-1]:
        L = layers.CuDNNLSTM(nc, return_sequences=True)
        H1 = L(H1)
        H2 = L(H2)

    H = layers.Concatenate(axis=1)([H1,H2])
    H = layers.CuDNNLSTM(n_cells_disc[-1])(H)

    H2 = layers.CuDNNLSTM(n_cells_disc[-1])(H2)

    H = layers.Concatenate()([H,H2])

    H = layers.Dense(128)(H)
    H = layers.LeakyReLU()(H)
    H = layers.Dense(1, activation='sigmoid', use_bias=False)(H)
    disc_output = H

    discriminator = Model([disc_input1,disc_input2], disc_output, name='discriminator')
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')

    discriminator.trainable = False

    gan_output = discriminator([input_seq,generator_output])
    gan = Model([input_seq, input_noise], gan_output, name='gan')

    gan.hp_xe_weight = K.variable(1.0)
    gan.hp_rigidity_weight = K.variable(1.0)

    y_ones = K.ones((K.shape(gan_output)[0],1))
    xe_loss = K.mean(K.binary_crossentropy(y_ones, gan_output), axis=-1)
    gan.add_loss(gan.hp_xe_weight*K.mean(xe_loss))

    if rigid_edges:
        rigidity_loss = None
        for i,j,m,v in rigid_edges:
            a = generator_output[:,:,i]
            b = generator_output[:,:,j]
            D = K.sum(K.square(a-b), axis=-1)
            l = K.square(D - m) / v
            if rigidity_loss is None:
                rigidity_loss = l
            else:
                rigidity_loss += l
        rigidity_loss /= len(rigid_edges)
        rigidity_loss = K.mean(rigidity_loss, axis=-1)
        gan.add_loss(gan.hp_rigidity_weight*K.mean(rigidity_loss))

    gan.compile(optimizer='adam')
    discriminator.trainable = True

    return generator, discriminator, gan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotate", action='store_true', help="Apply rotations during training")
    parser.add_argument("--lr-gen", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--lr-disc", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--rigidity", type=float, default=1e-4, help="The rigidity loss weight")
    parser.add_argument("--n-edges", type=int, default=140, help="Number of edge constraints")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--n-prompt", type=int, default=24, help="Number of frames in the prompt sequence")
    parser.add_argument("--n-pred", type=int, default=4, help="Number of future frames to predict")
    parser.add_argument("--noise-dim", type=int, default=128, help="Noise dimensionality")
    parser.add_argument("--gpu", type=str, help="GPU selector (on gatsby)")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--force", action='store_true', help="Force overwrite directories")
    parser.add_argument("--data", default="carrie-10-mins.npy", help="Data to train on")
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

    dirname = 'gan_rot%d_b%d_lrg%g_lrd%g_rw%g'%(args.rotate, args.batch_size, args.lr_gen, args.lr_disc, args.rigidity)
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
    X, vtx_pairs = load_data(args.data)

    n_vtx = X.shape[1]
    q5_mean, q95_mean = np.quantile(np.array(vtx_pairs)[:,2], [0.5,0.95])
    q5_var, q95_var = np.quantile(np.array(vtx_pairs)[:,3], [0.5,0.95])

    epochs = args.epochs
    batch_size = args.batch_size
    do_rotation = args.rotate
    n_prompt = args.n_prompt
    n_pred = args.n_pred
    noise_dim = args.noise_dim
    n_edges = args.n_edges

    gen, disc, gan = mk_model(X,
            n_prompt=n_prompt,
            n_pred=n_pred,
            noise_dim=noise_dim,
            rigid_edges=vtx_pairs[:n_edges])

    K.set_value(gan.optimizer.lr, args.lr_gen)
    K.set_value(disc.optimizer.lr, args.lr_disc)

    d_losses = []
    g_losses = []

    nbatches = (len(X)-batch_size*n_prompt-n_pred)//batch_size

    y_real = np.ones(batch_size)
    y_fake = np.zeros(batch_size)
    print("Begin training %d epochs, with %d batches per epoch."%(epochs, nbatches))
    for iepoch in range(epochs):
        offsets = np.random.choice(len(X)-batch_size*n_prompt-n_pred, replace=False, size=(nbatches,batch_size))
        
        d_loss = 0
        g_loss = 0
        for ibatch in range(nbatches):
            batch_idxs = offsets[ibatch].repeat(n_prompt).reshape(batch_size,n_prompt) + np.arange(n_prompt)
            truth_idxs = offsets[ibatch].repeat(n_pred).reshape(batch_size,n_pred) + np.arange(n_pred) + n_prompt
            
            x1_real = X[batch_idxs]
            x2_real = X[truth_idxs]
            if do_rotation:
                theta = 2*np.pi*np.random.rand()
                c,s = np.cos(theta), np.sin(theta)
                rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                x1_real = rot.dot(x1_real.transpose((0,1,3,2))).transpose((1,2,3,0))
                x2_real = rot.dot(x2_real.transpose((0,1,3,2))).transpose((1,2,3,0))
            
            noise = np.random.normal(size=(batch_size,noise_dim))
            x2_fake = gen.predict([x1_real, noise])
            
            disc.trainable = True
            d_loss += 0.5*disc.train_on_batch([x1_real, x2_real], y_real)
            d_loss += 0.5*disc.train_on_batch([x1_real, x2_fake], y_fake)
            
            if do_rotation:
                theta = 2*np.pi*np.random.rand()
                c,s = np.cos(theta), np.sin(theta)
                rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                x1_real = rot.dot(x1_real.transpose((0,1,3,2))).transpose((1,2,3,0))

            noise = np.random.normal(size=(batch_size,noise_dim))
            
            disc.trainable = False
            g_loss += gan.train_on_batch([x1_real, noise], None)
            
        d_losses.append(d_loss/nbatches)
        g_losses.append(g_loss/nbatches)
        print("Epoch %d/%d: L(d)=%.2e L(g)=%.2e" % (iepoch, epochs, d_losses[-1], g_losses[-1]))
        sys.stdout.flush()

        if iepoch%10==0:
            fig = plt.figure(0, figsize=(9,4))
            plt.clf()
            plt.subplot(1,2,1)
            plt.plot(d_losses, label='disc')
            plt.plot(g_losses, label='gen')
            plt.legend()
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('loss', fontsize=14)
            plt.yscale('log')
            plt.subplot(1,2,2)
            plt.plot(d_losses[:-128], label='disc')
            plt.plot(g_losses[:-128], label='gen')
            plt.legend()
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('loss', fontsize=14)
            fname = os.path.join(output_path, 'losses.png')
            plt.savefig(fname)
            print("Saved losses to", fname)

            fig = plt.figure(1, figsize=(9,9))
            plt.clf()
            examples = gen.predict([x1_real[:4], np.random.normal(size=(4,args.noise_dim))])
            for iplt in range(4):
                ax = plot_stick(examples[iplt,0], vtx_pairs[:n_edges], fig=fig, subplot=(2,2,iplt+1))
                plot_pose(examples[iplt,0], ax=ax, center=True)
            plt.suptitle("Epoch %d"%iepoch)
            fname = os.path.join(output_path, 'poses_e%04dpng'%iepoch)
            plt.savefig(fname)
            print("Saved poses to", fname)

            iprompt = np.random.randint(len(X)-batch_size*n_prompt-1)
            ngen = 3*n_prompt

            gen_seq = np.zeros((n_prompt+ngen, X.shape[1], X.shape[2]))
            gen_seq[:n_prompt] = X[iprompt:iprompt+n_prompt]
            dm = np.zeros((n_vtx,n_vtx))
            dv = np.zeros_like(dm)
            for i,j in it.combinations(range(n_vtx), r=2):
                dm[i,j] = dm[j,i] = np.sqrt((gen_seq[n_prompt:,i]-gen_seq[n_prompt:,j])**2).sum(axis=-1).mean()
                dv[i,j] = dv[j,i] = np.sqrt((gen_seq[n_prompt:,i]-gen_seq[n_prompt:,j])**2).sum(axis=-1).var(ddof=1)

            fig = plt.figure(2, figsize=(9,4))
            plt.subplot(1,2,1)
            plt.imshow(dm, vmin=q5_mean, vmax=q95_mean)
            plt.title('mean')
            plt.subplot(1,2,2)
            plt.imshow(dv, vmin=q5_var, vmax=q95_var)
            plt.title('var')
            plt.suptitle('Epoch %d'%iepoch)
            fname = os.path.join(output_path, 'rigidity_e%04dpng'%iepoch)
            plt.savefig(fname)
            print("Saved rigidity scores to", fname)
            sys.stdout.flush()


        if iepoch%10==0 or iepoch==(args.epochs-1):
            for m,mname in ( (gen,'gen'),(disc,'disc'),(gan,'gan'),):
                model_file = os.path.join(output_path, 'model-%s-latest.h5'%mname)
                weights_file = os.path.join(output_path, 'weights-%s-latest.h5'%mname)
                m.save(model_file)
                m.save_weights(weights_file)
                print("Saved model to", model_file)
                print("Saved weights to", weights_file)
            sys.stdout.flush()
        if iepoch%100==0:
            for m,mname in ( (gen,'gen'),(disc,'disc'),(gan,'gan'),):
                model_file = os.path.join(output_path, 'model-%s-e%04d.h5'%(mname,iepoch))
                weights_file = os.path.join(output_path, 'weights-%s-e%04d.h5'%(mname,iepoch))
                m.save(model_file)
                m.save_weights(weights_file)
                print("Saved model to", model_file)
                print("Saved weights to", weights_file)
            sys.stdout.flush()
