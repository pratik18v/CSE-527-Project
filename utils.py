import random
import cPickle
import numpy as np

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

PIXELS = 32
PAD_CROP = 4
PAD_PIXELS = PIXELS + (PAD_CROP * 2)
imageSize = PIXELS * PIXELS
num_features = imageSize * 3

# ##################### Load data from CIFAR-10 dataset #######################
# this code assumes the cifar dataset from 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# has been extracted in a 'data' folder within the working directory

def load_pickle_data_cv():

    """
    fo_1 = open('data/cifar-10-batches-py/data_batch_1', 'rb')
    fo_2 = open('data/cifar-10-batches-py/data_batch_2', 'rb')
    fo_3 = open('data/cifar-10-batches-py/data_batch_3', 'rb')
    fo_4 = open('data/cifar-10-batches-py/data_batch_4', 'rb')
    fo_5 = open('data/cifar-10-batches-py/data_batch_5', 'rb')
    dict_1 = cPickle.load(fo_1)
    fo_1.close()
    dict_2 = cPickle.load(fo_2)
    fo_2.close()
    dict_3 = cPickle.load(fo_3)
    fo_3.close()
    dict_4 = cPickle.load(fo_4)
    fo_4.close()
    dict_5 = cPickle.load(fo_5)
    fo_5.close()

    #Take only 10% of dataset
    data_1 = dict_1['data'][0:5000,:]
    data_2 = dict_2['data'][0:5000,:]
    data_3 = dict_3['data'][0:5000,:]
    data_4 = dict_4['data'][0:5000,:]
    data_5 = dict_5['data'][0:5000,:]
    labels_1 = dict_1['labels'][0:5000]
    labels_2 = dict_2['labels'][0:5000]
    labels_3 = dict_3['labels'][0:5000]
    labels_4 = dict_4['labels'][0:5000]
    labels_5 = dict_5['labels'][0:5000]


    X_train = np.vstack((data_1, data_2, data_3, data_4, data_5))
    y_train = np.hstack((labels_1, labels_2, labels_3, labels_4, labels_5)).astype('int32')


    X_train, y_train = shuffle(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    X_train = X_train.reshape(X_train.shape[0], 3, PIXELS, PIXELS).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 3, PIXELS, PIXELS).astype('float32')

    #printing array sizes
    print(X_train.shape)
    print(X_test.shape)

    # subtract per-pixel mean
    pixel_mean = np.mean(X_train, axis=0)
    print pixel_mean
    np.save('data/pixel_mean.npy', pixel_mean)
    X_train -= pixel_mean
    X_test -= pixel_mean

    #saving files to extract features
    np.save('saved/X_train.npy', X_train)
    np.save('saved/X_test.npy', X_test)
    np.save('saved/y_train.npy', y_train)
    np.save('saved/y_test.npy', y_test)
    """

    #loading already saved data
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')


    """
    Xtrn_sift1 = np.load('train_sift_descriptors1.npy')
    Xtrn_sift2 = np.load('train_sift_descriptors2.npy')
    Xtrn_sift3 = np.load('train_sift_descriptors3.npy')
    Xtrn_sift4 = np.load('train_sift_descriptors4.npy')
    Xtrn_sift5 = np.load('train_sift_descriptors5.npy')
    Xtrn_sift6 = np.load('train_sift_descriptors6.npy')
    Xtrn_sift7 = np.load('train_sift_descriptors7.npy')
    Xtrn_sift8 = np.load('train_sift_descriptors8.npy')
    Xtrn_sift9 = np.load('train_sift_descriptors9.npy')

    Xtst_feat = np.load('test_sift_descriptors.npy')

    Xtrn_sift1 = Xtrn_sift1.reshape(5000, 128, PIXELS, PIXELS).astype('float32')
    Xtrn_sift2 = Xtrn_sift2.reshape(5000, 128, PIXELS, PIXELS).astype('float32')
    Xtrn_sift3 = Xtrn_sift3.reshape(5000, 128, PIXELS, PIXELS).astype('float32')
    Xtrn_sift4 = Xtrn_sift4.reshape(5000, 128, PIXELS, PIXELS).astype('float32')
    Xtrn_sift5 = Xtrn_sift5.reshape(5000, 128, PIXELS, PIXELS).astype('float32')
    Xtrn_sift6 = Xtrn_sift6.reshape(5000, 128, PIXELS, PIXELS).astype('float32')
    Xtrn_sift7 = Xtrn_sift7.reshape(5000, 128, PIXELS, PIXELS).astype('float32')
    Xtrn_sift8 = Xtrn_sift8.reshape(5000, 128, PIXELS, PIXELS).astype('float32')
    Xtrn_sift9 = Xtrn_sift9.reshape(5000, 128, PIXELS, PIXELS).astype('float32')



    Xtrn_feat = np.vstack((Xtrn_sift1,Xtrn_sift2,Xtrn_sift3,Xtrn_sift4,Xtrn_sift5,Xtrn_sift6,Xtrn_sift7,Xtrn_sift8,Xtrn_sift9))

    #print Xtrn_sift.shape, Xtst_sift.shape
    """

    Xtrn_feat = np.load('data_and_features/train_hog.npy')
    Xtst_feat = np.load('data_and_features/test_hog.npy')

    X_train = np.concatenate((X_train,Xtrn_feat),axis=1)
    X_test = np.concatenate((X_test,Xtst_feat),axis=1)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)


    #print "Train set size: {}, Test set size: {}".format(X_train.shape, X_test.shape)

    #fac_train = 22500
    #fac_test = 2500

    return X_train, X_test, y_train, y_test
    #return X_train[0:fac_train,:,:,:], X_test[0:fac_test,:,:,:], y_train[0:fac_train], y_test[0:fac_test]

def load_pickle_data_test():
    fo_test = open('data/cifar-10-batches-py/test_batch', 'rb')
    dict_test = cPickle.load(fo_test)
    fo_test.close()
    test_X = dict_test['data']
    test_y = dict_test['labels']
    test_y = np.hstack(test_y).astype('int32')

    test_X = test_X.reshape(test_X.shape[0], 3, PIXELS, PIXELS).astype('float32')

    pixel_mean = np.load('data/pixel_mean.npy')
    test_X -= pixel_mean

    #test_feat = np.load('test_data_hog_descriptors.npy')
    #test_feat = np.load('test_data_sift_descriptors.npy')

    #test_X = np.concatenate((test_X,test_feat), axis=1)
    #test_X = test_X.astype(np.float32)

    return test_X, test_y

def batch_iterator_train_crop_flip(data, y, batchsize, train_fn):
    '''
    Data augmentation batch iterator for feeding images into CNN.
    Pads each image with 4 pixels on every side.
    Randomly crops image with original image shape from padded image. Effectively translating it.
    Flips image lr with probability 0.5.
    '''
    n_samples = data.shape[0]
    # Shuffles indicies of training data, so we can draw batches from random indicies instead of shuffling whole data
    indx = np.random.permutation(xrange(n_samples))
    loss = []
    acc_train = 0.
    for i in range((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[indx[sl]]
        y_batch = y[indx[sl]]

        # pad and crop settings
        trans_1 = random.randint(0, (PAD_CROP*2))
        trans_2 = random.randint(0, (PAD_CROP*2))
        crop_x1 = trans_1
        crop_x2 = (PIXELS + trans_1)
        crop_y1 = trans_2
        crop_y2 = (PIXELS + trans_2)

        # flip left-right choice
        flip_lr = random.randint(0,1)

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.copy(X_batch)

        # for each image in the batch do the augmentation
        for j in range(X_batch.shape[0]):
            # for each image channel
            for k in range(X_batch.shape[1]):
                # pad and crop images
                img_pad = np.pad(X_batch_aug[j,k], pad_width=((PAD_CROP,PAD_CROP), (PAD_CROP,PAD_CROP)), mode='constant')
                X_batch_aug[j,k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

                # flip left-right if chosen
                if flip_lr == 1:
                    X_batch_aug[j,k] = np.fliplr(X_batch_aug[j,k])

        # fit model on each batch
        loss.append(train_fn(X_batch_aug, y_batch))

    return np.mean(loss)

def batch_iterator_valid(data_test, y_test, batchsize, valid_fn):
    '''
    Batch iterator for fine tuning network, no augmentation.
    '''
    n_samples_valid = data_test.shape[0]
    loss_valid = []
    acc_valid = []
    for i in range((n_samples_valid + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch_test = data_test[sl]
        y_batch_test = y_test[sl]

        loss_vv, acc_vv = valid_fn(X_batch_test, y_batch_test)
        loss_valid.append(loss_vv)
        acc_valid.append(acc_vv)

    return np.mean(loss_valid), np.mean(acc_valid)



def main():
    load_pickle_data_cv()


if __name__ == "__main__":
    main()
