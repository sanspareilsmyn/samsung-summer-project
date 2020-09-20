#HoG
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
from config import *

#def hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm=None, visualize=False, visualise=None, transform_sqrt=False,feature_vector=True, multichannel=None):
    
def extract_features():
    des_type = 'HOG'

    # If feature directories don't exist, create them
    if not os.path.isdir(c1_feat_ph):
        os.makedirs(c1_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(c2_feat_ph):
        os.makedirs(c2_feat_ph)

    print ("Calculating the descriptors for the positive samples and saving them")
    for im_path in glob.glob(os.path.join(c1_im_path, "*.jpg")):
        #print im_path
        im = imread(im_path,as_grey=True)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(c1_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print ("Positive features saved in {}".format(c1_feat_ph))

    print ("Calculating the descriptors for the negative samples and saving them")
    for im_path in glob.glob(os.path.join(c2_im_path, "*.jpg")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im,  orientations, pixels_per_cell, cells_per_block)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(c2_feat_ph, fd_name)
    
        joblib.dump(fd, fd_path)
    print ("Negative features saved in {}".format(c2_feat_ph))

    print ("Completed calculating features from training images")

if __name__=='__main__':
    extract_features()
