#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
import pickle
import itertools as it
import time
import signal

import keras.backend as K
from keras import layers
from keras.models import Model
from keras.utils import to_categorical
import tensorflow as tf

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

#from utils.plot import plot_pose, plot_stick
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import juggle_axes

def plot_pose(x, ax=None, fig=None, subplot=None,
              lim=(-0.75,0.75), center=False):
    if fig is None:
        if ax:
            fig = ax.get_figure()
        else:
            fig = plt.figure()
    if not ax:
        if subplot:
            ax = fig.add_subplot(*subplot, projection='3d')
        else:
            ax = p3.Axes3D(fig)
    if center:
        lim = x.min(), x.max()
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_zlim(*lim)
    ax.scatter(x[:,0], x[:,1], x[:,2], marker='.')
    return ax

def plot_stick(x, edges, fig=None, ax=None, subplot=None,
               lim=(-0.75,0.75), center=False):
    if fig is None:
        if ax:
            fig = ax.get_figure()
        else:
            fig = plt.figure()
    if ax is None:
        if subplot:
            ax = fig.add_subplot(*subplot, projection='3d')
        else:
            ax = p3.Axes3D(fig)
    if center:
        lim = x.min(), x.max()
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_zlim(*lim)
    for e in edges:
        ax.plot(np.linspace(x[e[0],0],x[e[1],0],10),
                np.linspace(x[e[0],1],x[e[1],1],10),
                np.linspace(x[e[0],2],x[e[1],2],10)
               )
    return ax

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

    lstm = layers.CuDNNLSTM
    #lstm = layers.LSTM
    for nc in n_cells:
        H = lstm(nc, return_sequences=True)(H)
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
        L = lstm(nc, return_sequences=True)
        H1 = L(H1)
        H2 = L(H2)

    H = layers.Concatenate(axis=1)([H1,H2])
    H = lstm(n_cells_disc[-1])(H)

    H2 = lstm(n_cells_disc[-1])(H2)

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

class Log:
    def __init__(self, fpath):
        self._file = open(fpath, 'w')
    def write(self, msg):
        self._file.write(msg)
        self._file.flush()

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
    parser.add_argument("--do-plots", action='store_true', help="Save plots to log dir")
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

    dirname = 'gan_rot%d_b%d_np%d_lrg%g_lrd%g_rw%g'%(args.rotate, args.batch_size, args.n_pred, args.lr_gen, args.lr_disc, args.rigidity)
    output_path = os.path.join(args.path, dirname)


    try:
        os.makedirs(output_path)
    except OSError as e:
        import errno
        if e.errno == errno.EEXIST:
            if args.force:
                print("Warning -- writing results to existing path:", output_path)
                sys.stdout.flush()
            else:
                print("Skipping run because of existing path:", output_path)
                sys.exit()
        else:
            raise

    status_file = Log(os.path.join(output_path, 'status'))
    status_file.write("init\n")

    abort_now = False
    abort_run = False
    abort_reason = None
    training = False
    def signal_handler(sig, frame):
        print("***Signaled!", sig)
        status_file.write("signal (%s)\n"%sig)
        abort_now = True
        abort_run = True
        abort_reason = "signal"
        if not training:
            sys.exit()

    except_hook_default = sys.excepthook
    def except_hook(exctype, value, tb):
        status_file.write("exception (%s)\n"%exctype)
        except_hook_default(exctype, value, tb)
    sys.excepthook = except_hook

    #signal.signal(signal.SIGINT, signal_handler)
    #signal.signal(signal.SIGTERM, signal_handler)
    for i in [x for x in dir(signal) if x.startswith("SIG")]:
        try:
            signum = getattr(signal,i)
            signal.signal(signum, signal_handler)
        except:
            pass

    with open(os.path.join(output_path, 'config.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    if 'SLURM_JOB_ID' in os.environ:
        with open(os.path.join(output_path, 'jid.%s'%os.environ['SLURM_JOB_ID']), 'w') as f:
            pass
        with open(os.path.join(output_path, 'slurm_job'), 'w') as f:
            f.write(os.environ['SLURM_JOB_ID']+"\n")
    else:
        with open(os.path.join(output_path, 'pid.%d'%os.getpid()), 'w') as f:
            pass

    print("Loading data")
    sys.stdout.flush()
    X, vtx_pairs = load_data(args.data)

    n_vtx = X.shape[1]
    q5_mean, q95_mean = np.percentile(np.array(vtx_pairs)[:,2], [5,95])
    q5_var, q95_var = np.percentile(np.array(vtx_pairs)[:,3], [5,95])

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
    K.set_value(gan.hp_rigidity_weight, args.rigidity)

    d_losses = []
    g_losses = []

    nbatches = (len(X)-batch_size*n_prompt-n_pred)//batch_size

    subplots = {}
    y_real = np.ones(batch_size)
    y_fake = np.zeros(batch_size)
    print("Begin training %d epochs, with %d batches per epoch."%(epochs, nbatches))
    training = True
    sys.stdout.flush()
    tstart = time.time()
    status_file.write('training\n')
    loss_file = Log(os.path.join(output_path, 'losses.log'))
    for iepoch in range(epochs):
        estart = time.time()
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
            if abort_now:
                break
            
        d_losses.append(d_loss/(ibatch+1))
        g_losses.append(g_loss/(ibatch+1))
        loss_file.write('%g\t%g\n'%(d_losses[-1], g_losses[-1]))
        print("Epoch %d/%d (%03.2fs): L(d)=%.2e L(g)=%.2e" % (iepoch, epochs, time.time()-estart, d_losses[-1], g_losses[-1]))
        sys.stdout.flush()

        if iepoch>10 and np.mean(d_losses[-10:])<1e-5:
            abort_run = True
            abort_reason = "ldisc"

        if iepoch%10==0 or abort_run:
            if args.do_plots:
                fig = plt.figure(0, figsize=(9,4))
                plt.clf()
                plt.subplot(1,2,1)
                xepochs = np.arange(1, len(d_losses)+1)
                plt.plot(xepochs,d_losses, label='disc')
                plt.plot(xepochs,g_losses, label='gen')
                plt.legend()
                plt.xlabel('epoch', fontsize=14)
                plt.ylabel('loss', fontsize=14)
                plt.yscale('log')
                plt.subplot(1,2,2)
                plt.plot(xepochs[-64:],d_losses[-64:], label='disc')
                plt.plot(xepochs[-64:],g_losses[-64:], label='gen')
                plt.legend()
                plt.xlabel('epoch', fontsize=14)
                plt.ylabel('loss', fontsize=14)
                fname = os.path.join(output_path, 'losses.png')
                plt.savefig(fname)
                #np.save(os.path.join(output_path, 'losses-g.npy'), g_losses)
                #np.save(os.path.join(output_path, 'losses-d.npy'), d_losses)

            noise = np.random.normal(size=(4,args.noise_dim))
            samples = gen.predict([x1_real[:4], noise])

            if args.do_plots:
                fig = plt.figure(1, figsize=(9,9))
                plt.clf()
                for iplt in range(4):
                    ax = plot_stick(samples[iplt,0], vtx_pairs[:n_edges], fig=fig, subplot=(2,2,iplt+1))
                    plot_pose(samples[iplt,0], ax=ax, center=True)
                plt.suptitle("Epoch %d"%iepoch)
                fname = os.path.join(output_path, 'poses_e%04d.png'%iepoch)
                plt.savefig(fname)
            np.savez(os.path.join(output_path, 'samples_e%04d.npz'%iepoch), prompt=x1_real[:4], noise=noise, pred=samples, edges=vtx_pairs[:n_edges])

            dm = np.zeros((n_vtx,n_vtx))
            dv = np.zeros_like(dm)
            for i,j in it.combinations(range(n_vtx), r=2):
                dist = np.sqrt(np.sum((samples[:,:,i]-samples[:,:,j])**2, axis=-1))
                dm[i,j] = dm[j,i] = dist.mean()
                dv[i,j] = dv[j,i] = dist.var(ddof=1)
            np.savez(os.path.join(output_path, 'rigidity_e%04d.npz'%iepoch), mean=dm, var=dv)

            if args.do_plots:
                fig = plt.figure(2, figsize=(9,4))
                plt.subplot(1,2,1)
                plt.imshow(dm, vmin=q5_mean, vmax=q95_mean)
                plt.title('mean')
                plt.subplot(1,2,2)
                plt.imshow(dv, vmin=q5_var, vmax=q95_var)
                plt.title('var')
                plt.suptitle('Epoch %d'%iepoch)
                fname = os.path.join(output_path, 'rigidity_e%04d.png'%iepoch)
                plt.savefig(fname)
                print("Saved plots to", output_path)
                sys.stdout.flush()

            print("Saved samples to", output_path)


        if iepoch%10==0 or iepoch==(args.epochs-1) or abort_run:
            for m,mname in ( (gen,'gen'),(disc,'disc'),(gan,'gan'),):
                model_file = os.path.join(output_path, 'model-%s-latest.h5'%mname)
                weights_file = os.path.join(output_path, 'weights-%s-latest.h5'%mname)
                m.save(model_file)
                m.save_weights(weights_file)
            print("Checkpoint model at",output_path)
            sys.stdout.flush()
        if iepoch%100==0 or abort_run:
            for m,mname in ( (gen,'gen'),(disc,'disc'),(gan,'gan'),):
                model_file = os.path.join(output_path, 'model-%s-e%04d.h5'%(mname,iepoch))
                weights_file = os.path.join(output_path, 'weights-%s-e%04d.h5'%(mname,iepoch))
                m.save(model_file)
                m.save_weights(weights_file)
            print("Saved model for e%04d at %s"%(iepoch,output_path))
            sys.stdout.flush()

        if abort_run:
            status_file.write('abort (%s)\n'%abort_reason)
            if abort_reason=='ldisc':
                print("***Exiting early due to low discriminator losses.")
            elif abort_reason=='signal':
                print("***Exiting early due to signal.")
            else:
                print("***Exiting early due to unknown reason.")
            break

    if iepoch==(epochs-1):
        # we actually made it to the end!
        status_file.write('complete\n')

    print("Training complete after %d epochs. Total time: %ds"%(iepoch, time.time()-tstart))
    status_file.write('finished\n')
