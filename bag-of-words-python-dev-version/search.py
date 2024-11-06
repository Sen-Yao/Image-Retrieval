#!/usr/local/bin/python2.7
#python search.py -i dataset/testing/all_souls_000000.jpg

import argparse as ap
import cv2
import numpy as np
import joblib
from sklearn import preprocessing
from rootsift import RootSIFT

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required="True")
args = vars(parser.parse_args())

# Get query image path
image_path = args["image"]

# Load the classifier, class names, scaler, number of clusters and vocabulary 
im_features, image_paths, idf, numWords, voc = joblib.load("bag-of-words.pkl")

# Load inverted index
inverted_index = joblib.load("inverted_index.pkl")

# Create feature extraction and keypoint detector objects
fea_det = cv2.xfeatures2d.SIFT_create()

# Read and resize the query image
im = cv2.imread(image_path)
im_size = im.shape
im = cv2.resize(im, (im_size[1] // 4, im_size[0] // 4))

# Extract keypoints and descriptors for the query image
kpts_query, des_query = fea_det.detectAndCompute(im, None)

# Prepare to store relevant images
relevant_images = set()

# Create a Vocabulary Tree
num_levels = 5  # Number of levels in the tree
branching_factor = 10  # Number of branches per node
vocab_tree = cv2.VocabularyTree(branching_factor, num_levels)

# Assuming you have a list of all descriptors from your training images
# You would typically load these from your database
# For demonstration, let's say you have a list of descriptors
# descriptors_list = [...]  # List of all descriptors from training images

# Build the Vocabulary Tree with the descriptors
# vocab_tree.build(descriptors_list)

# Compute BoW features for the query image
test_features = np.zeros((1, numWords), "float32")
words, distance = vocab_tree.predict(des_query)  # Use the Vocabulary Tree to predict words
for w in words:
    test_features[0][w] += 1

# Perform TF-IDF vectorization and L2 normalization
test_features = test_features * idf
test_features = preprocessing.normalize(test_features, norm='l2')

# Use inverted index to find relevant images
for w in words:
    if w in inverted_index:
        relevant_images.update(inverted_index[w])

if not relevant_images:
    print("No related images found.")
    exit()

relevant_images = list(relevant_images)

# Prepare to store the scores and matched images
scores = []
matched_images = []

# Iterate through relevant images and perform spatial verification
for img_path in relevant_images:
    # Read the relevant image
    img_relevant = cv2.imread(img_path)
    img_relevant = cv2.resize(img_relevant, (img_relevant.shape[1] // 4, img_relevant.shape[0] // 4))

    # Extract keypoints and descriptors for the relevant image
    kpts_relevant, des_relevant = fea_det.detectAndCompute(img_relevant, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_query, des_relevant)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Apply RANSAC to find good matches
    if len(matches) > 10:  # Ensure there are enough matches
        src_pts = np.float32([kpts_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts_relevant[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        # Count inliers
        inliers = mask.ravel().tolist()
        num_inliers = inliers.count(1)

        # Store the score based on the number of inliers
        scores.append(num_inliers)
        matched_images.append(img_path)

# Sort images based on scores
rank_ID = np.argsort(-np.array(scores))

# Visualize the results
for i in range(min(16, len(rank_ID))):
    print(str(matched_images[rank_ID[i]]))
