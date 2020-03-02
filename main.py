#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from scipy import signal
from sklearn.utils import shuffle
import numpy as np
import keras
from keras.models import *
from keras.utils import *
from keras.layers import *
from keras.layers.merge import _Merge
from keras.optimizers import *
from keras.preprocessing.image import *
from keras import backend as K
from functools import partial
from utils import *

### most codes are adopted from https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

# load dataset
X_train=np.load("Zxx_train.npy")
Y_train=np.load("Zyy_train_poly.npy")
x_test=np.load("Zxx_test.npy")
y_test=np.load("Zyy_test_poly.npy")

# filename for saved optimized generator
generator_filename = 'generator_poly.h5'

seed=754265
np.random.seed(seed)

num_valid=X_train.shape[0]//10 # 208 out of 2080 (10% as valid --> train:valid=9:1)
x_train=X_train[num_valid:] 
y_train=Y_train[num_valid:]
x_valid=X_train[:num_valid]
y_valid=Y_train[:num_valid]

# finally shuffle here not to mixup patients for training and validation set
x_train, y_train=shuffle(x_train, y_train, random_state=seed)
x_valid, y_valid=shuffle(x_valid, y_valid, random_state=seed)
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

#%% hyperparameters
batch_size = 64 
TRAINING_RATIO = 1 
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
plot_epoch_interval=2500
valid_epoch_interval=1000
epochs=5000
df=16
nperseg=14
img_shape=x_train.shape[1:]
channels=x_train.shape[3] # 2
#%%
def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed",
    and outputs images of size 28x28x1."""
    input_img = Input(shape=img_shape)
    activation_function='elu'    
    
    x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(input_img)
    x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(encoded)
    x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(channels, (3, 3), activation='linear', padding='same')(x)
    
    model=Model(input_img, decoded)
    model.summary()
    return model 



def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be
    used in the discriminator."""
    def d_layer(layer_input, filters, f_size=3, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_x = Input(shape=img_shape)
    img_y = Input(shape=img_shape)

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([img_y, img_x])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(channels, kernel_size=3, strides=1, padding='same')(d4) 

    model=Model([img_x, img_y], validity)
    model.summary()
    return model

def load_batch(x,y,batch_size):
        n_batches=len(x)//batch_size
        for i in range(n_batches-1):
            x_batch, y_batch = x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
            xs_batch, ys_batch = [],[]
            for idx in range(batch_size):
                if np.random.random() > 0.5:
                    x_batch[idx] = np.fliplr(x_batch[idx])
                    y_batch[idx] = np.fliplr(y_batch[idx])
                xs_batch.append(x_batch[idx])
                ys_batch.append(y_batch[idx])
            xs_batch=np.stack(xs_batch)
            ys_batch=np.stack(ys_batch)
#            print(xs_batch.shape, ys_batch.shape)
            
            yield i, (xs_batch, ys_batch)
        
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

# Now we initialize the generator and discriminator.
generator = make_generator()
discriminator = make_discriminator()

# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within
# it. As such, it won't cause problems if we later set discriminator.trainable = True
# for the discriminator_model, as long as we compile the generator_model first.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
img_x_D = Input(shape=img_shape)
generated_y = generator(img_x_D)
discriminator_layers_for_generator = discriminator([img_x_D, generated_y])
generator_model = Model(inputs=[img_x_D],
                        outputs=[discriminator_layers_for_generator, generated_y])
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0002, beta_1=0.5, beta_2=0.9),
                        loss=[wasserstein_loss, 'mae'],
                        loss_weights=[1,1])

# Now that the generator_model is compiled, we can make the discriminator
# layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random
# noise seeds as input. The noise seed is run through the generator model to get
# generated images. Both real and generated images are then run through the
# discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
img_x_G = Input(shape=img_shape)
img_y_G = Input(shape=img_shape)
generated_y_for_discriminator = generator(img_x_G)
discriminator_output_from_generated_y = discriminator([img_x_G, generated_y_for_discriminator])
discriminator_output_from_real_y = discriminator([img_x_G, img_y_G])


averaged_samples = RandomWeightedAverage()([img_y_G,
                                            generated_y_for_discriminator])

# We then run these samples through the discriminator as well. Note that we never
# really use the discriminator output for these samples - we're only running them to
# get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator([img_x_G, averaged_samples])

# The gradient penalty loss function requires the input averaged samples to get
# gradients. However, Keras loss functions can only have two arguments, y_true and
# y_pred. We get around this by making a partial() of the function with the averaged
# samples here.
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
# Functions need names or Keras will throw an error
partial_gp_loss.__name__ = 'gradient_penalty'

# Keras requires that inputs and outputs have the same number of samples. This is why
# we didn't concatenate the real samples and generated samples before passing them to
# the discriminator: If we had, it would create an output with 2 * batch_size samples,
# while the output of the "averaged" samples for gradient penalty
# would have only batch_size samples.

# If we don't concatenate the real and generated samples, however, we get three
# outputs: One of the generated samples, one of the real samples, and one of the
# averaged samples, all of size batch_size. This works neatly!
discriminator_model = Model(inputs=[img_x_G,
                                    img_y_G],
                            outputs=[discriminator_output_from_generated_y,
                                     discriminator_output_from_real_y,
                                     averaged_samples_out])
# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
# the real and generated samples, and the gradient penalty loss for the averaged samples
discriminator_model.compile(optimizer=Adam(0.0002, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss],
                            loss_weights=[1, 1, 1])
# We make three label vectors for training. positive_y is the label vector for real
# samples, with value 1. negative_y is the label vector for generated samples, with
# value -1. The dummy_y vector is passed to the gradient_penalty loss function and
# is not used.
positive_y = np.ones((batch_size, 1,1,channels), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((batch_size, 1,1,channels), dtype=np.float32)

for epoch in range(epochs):
    for batch_i, (image_batch_x, image_batch_y) in load_batch(x_train, y_train, batch_size):
        
        d_loss=discriminator_model.train_on_batch(
            [image_batch_x, image_batch_y],
            [negative_y, positive_y, dummy_y])
        g_loss=generator_model.train_on_batch([image_batch_x],
                                                         [positive_y, image_batch_y])

        generated_batch_y = generator.predict(image_batch_x)
 
        generator.save(generator_filename)
        
        # print the progress
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, epochs,
                                                                batch_i, len(x_train)//batch_size,
                                                                d_loss[0],
                                                                g_loss[0]))
        
        # print valid loss
        if epoch % valid_epoch_interval == 0 and epoch != 0:
            valid_g_loss=[]
            for batch_i, (image_valid_batch_x, image_valid_batch_y) in load_batch(x_valid, y_valid, batch_size):
                generated_valid_batch_y = generator.predict(image_valid_batch_x)
                v_g_loss=generator_model.train_on_batch([image_batch_x],
                                                         [positive_y, image_batch_y])
                valid_g_loss.append(v_g_loss)
            valid_loss=np.mean(valid_g_loss)
        
            # print the progress
            print ("[valid loss: %f]" % (valid_loss))
        # If at save interval => save generated image samples
        if epoch % plot_epoch_interval == 0 and epoch != 0:
#                    self.sample_images(epoch, batch_i)
            fig, ax=plt.subplots(3,3, figsize=(10,10))
            for idx in range(3):
                # plot time-signal curve
                t_rec, generated_y_plot=signal.istft(generated_batch_y[idx,:,:,0]+1j*generated_batch_y[idx,:,:,1], nperseg=nperseg)
                ax[idx,0].plot(t_rec, generated_y_plot, label='generated DSC')
                t_rec, image_y_plot=signal.istft(image_batch_y[idx,:,:,0]+1j*image_batch_y[idx,:,:,1], nperseg=nperseg)
                ax[idx,0].plot(t_rec, image_y_plot, label='DSC')
                t_rec, image_x_plot=signal.istft(image_batch_x[idx,:,:,0]+1j*image_batch_x[idx,:,:,1], nperseg=nperseg)
                ax[idx,0].plot(t_rec, image_x_plot, label='DCE')
                ax[idx,0].set_xlabel("Time-steps(TR)")
                ax[idx,0].set_ylabel("Signal intensity")

                # plot spectrogram
                ax[idx,1].imshow(np.sqrt(generated_batch_y[idx,:,:,0]**2+generated_batch_y[idx,:,:,1]**2))
                ax[idx,2].imshow(np.sqrt(image_batch_y[idx,:,:,0]**2+image_batch_y[idx,:,:,1]**2))
            plt.show()

#%% print PSNR
print("PSNR for train set:\n")
train_psnr = compute_total_PSNR(X_train, Y_train, batch_size)

print("PSNR for test set:\n")
test_psnr = compute_total_PSNR(x_test, y_test, batch_size)

