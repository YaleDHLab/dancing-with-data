# GAN
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import mean_squared_error

# params
latent_dim = 100
frame_shape = X[0].shape

##
# discriminator
##

def get_discriminator():
  '''Return a discriminator model'''
  d = Sequential()
  d.add(Flatten(input_shape=X.shape))
  d.add(Dense(512))
  d.add(LeakyReLU(alpha=0.2))
  d.add(Dense(256))
  d.add(LeakyReLU(alpha=0.2))
  d.add(Dense(1, activation='sigmoid'))

  # input a single frame and determine if it's real or generated
  frame = Input(shape=frame_shape)
  validity = d(frame)

  # return the actual discriminator
  return Model(frame, validitiy)

##
# generator
##

def get_generator():
  '''Return a generator model'''
  g = Sequential()
  g.add(Dense(256, input_dim=latent_dim))
  g.add(LeakyReLU(alpha=0.2))
  g.add(BatchNormalization(momentum=0.8))
  g.add(Dense(512))
  g.add(LeakyReLU(alpha=0.2))
  g.add(BatchNormalization(momentum=0.8))
  g.add(Dense(1024))
  g.add(LeakyReLU(alpha=0.2))
  g.add(BatchNormalization(momentum=0.8))
  g.add(Dense(np.prod(frame_shape), activation='tanh'))
  g.add(Reshape(frame_shape))

  noise = Input(shape=(latent_dim,))
  img = g(noise)
  
  # return the actual generator
  return Model(noise, img)

##
# Define the training function
##

def train(epochs=10000, batch_size=128, sample_interval=50):
  X_train = X
  # add a fourth dimension to X_train
  X_train = np.expand_dims(X_train, axis=3)
  # adversarial ground truth
  valid = np.ones((batch_size, 1))
  fake = np.zeros((batch_size, 1))
  
  for epoch in range(epochs):
    
    ##
    # train discriminator
    ##
    
    # select some random frames from the training data
    frame_indices = np.random.randint(0, X_train.shape[0], batch_size)
    frames = X_train[frame_indices]
    
    # get random noise with which to modulate the selected frames
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    
    # generate new frames
    gen_frames = generator.predict(noise)
    
    # train the discriminator on the frames
    d_loss_real = self.discriminator.train_on_batch(frames, valid)
    d_loss_fake = self.discriminator.train_on_batch(gen_frames, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    ##
    # train generator
    ##
    
    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
  
    g_loss = combined.train_on_batch(noise, valid)
    
    # print progress
    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    
    # periodically generate new samples
    if epoch % sample_interval == 0:
      sample_frames(epoch)
    

def sample_frames(epoch):
  r, c = 5, 5
  noise = np.random.normal(0, 1, (r * c, latent_dim))
  gen_frames = generator.predict(noise)
  
  # rescale 0-1
  gen_frames = 0.5 * gen_frames + 0.5
  
  # plot the gen_frames
  
##
# Build models
##

# center X features -1:1
X *= 2
X -= 1

# get the discriminator
discriminator = get_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminator.trainable = False # only train the generator

# get the generator
generator = get_generator()

# the generator takes random inputs and makes new frames
z = Input(shape=latent_dim)
frame = self.generator(z)
validity = discriminator(frame) # ask the discriminator to determine validity of frame

# stack the generator and discriminator into a combined model
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

#train(epochs=30000, batch_size=32, sample_interval=200)
