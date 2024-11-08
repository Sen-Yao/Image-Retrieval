#!/usr/local/bin/python2.7
# python search.py -i dataset/testing/all_souls_000000.jpg
import argparse as ap
import cv2
import numpy as np
# from sklearn.externals import joblib
import joblib
from scipy.cluster.vq import *
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np

from pylab import *
from PIL import Image
from rootsift import RootSIFT

class VocabularyTree:
    def __init__(self, max_depth=5, branching_factor=9):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.tree = None

    def fit(self, descriptors):
        self.tree = self._create_tree(descriptors, depth=0)

    def _create_tree(self, descriptors, depth):
        if depth >= self.max_depth or len(descriptors) == 0:
            return None

        n_clusters = min(self.branching_factor, len(descriptors))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(descriptors)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        children = []
        for i in range(n_clusters):
            child_descriptors = descriptors[labels == i]
            child_tree = self._create_tree(child_descriptors, depth + 1)
            children.append(child_tree)

        return {'center': cluster_centers, 'children': children}

    def get_vocabulary(self, max_size=None):
        vocabulary = []
        self._traverse_tree(self.tree, vocabulary)
        if max_size is not None:
            return np.vstack(vocabulary)[:max_size]  # 只返回前 max_size 个词汇
        return np.vstack(vocabulary)

    def _traverse_tree(self, node, vocabulary):
        if node is None:
            return
        vocabulary.append(node['center'])  # 这里假设 node['center'] 是一个数组
        for child in node['children']:
            self._traverse_tree(child, vocabulary)
# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required="True")
parser.add_argument("-sc", "--space_coefficient", help="space coefficient", type=float, default=0, required=True)

args = vars(parser.parse_args())

# Get query image path
image_path = args["image"]
# image_path='dataset/testing/all_souls_000000.jpg'
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
branching_factor = 9  # Number of branches per node
vocab_tree = VocabularyTree(branching_factor, num_levels)
vocab_tree.fit(des_query)

# Assuming you have a list of all descriptors from your training images
# You would typically load these from your database
# For demonstration, let's say you have a list of descriptors
# descriptors_list = [...]  # List of all descriptors from training images

# Build the Vocabulary Tree with the descriptors
# vocab_tree.build(descriptors_list)
voc1 = vocab_tree.get_vocabulary(max_size=1000)  # 限制 voc1 的大小为 1000
num=voc1.shape[0]
# Compute BoW features for the query image
test_features = np.zeros((1, num), "float32")
words, _ = vq(des_query, voc1)
# words, distance = vq(des_query,voc)
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
    src_pts = np.float32([kpts_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts_relevant[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Count inliers
    inliers = mask.ravel().tolist()
    num_inliers = inliers.count(1)

    # Store the score based on the number of inliers
    scores.append(args["space_coefficient"] * num_inliers + (1-args["space_coefficient"]) * len(matches))
    matched_images.append(img_path)

# Sort images based on scores
rank_ID = np.argsort(-np.array(scores))

# Visualize the results
figure()
gray()
subplot(5,4,1)
imshow(im[:,:,::-1])
axis('off')
for i in range(min(16, len(rank_ID))):
    # print(relevant_indices_list[ID])
    print(str(matched_images[rank_ID[i]]))
    img=Image.open(matched_images[rank_ID[i]])
    gray()
    subplot(5,4,i+5)
    imshow(img)
    axis('off')

show()  
# 用户反馈部分
# 用户反馈部分
print("Please provide feedback on the results:")
user_input = input("Best image index (space-separated): ")
best_index = int(user_input)

# 获取用户不喜欢的图片路径
unliked_image_path = matched_images[rank_ID[best_index]]
print("Best picture:", unliked_image_path)

scores[rank_ID[best_index]] = (scores[rank_ID[best_index]] + scores[rank_ID[0]]) / 2

# 特征提取
img_unliked = cv2.imread(unliked_image_path)
kpts_unliked, des_unliked = fea_det.detectAndCompute(img_unliked, None)

# 使用 BFMatcher 匹配描述子
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_unliked, des_unliked)

self_sim_score = len(matches)

# 准备一个字典来存储与用户不喜欢的图片相似的图片
similar_images = {}

# 遍历排名前32的相关图片
for idx in range(min(32, len(rank_ID))):
    img_path = matched_images[rank_ID[idx]]
    print(img_path)
    img_relevant = cv2.imread(img_path)
    img_relevant = cv2.resize(img_relevant, (img_relevant.shape[1] // 4, img_relevant.shape[0] // 4))
    kpts_relevant, des_relevant = fea_det.detectAndCompute(img_relevant, None)

    # 使用 BFMatcher 匹配描述子
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_unliked, des_relevant)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Apply RANSAC to find good matches
    src_pts = np.float32([kpts_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts_relevant[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Count inliers
    inliers = mask.ravel().tolist()
    num_inliers = inliers.count(1)

    similarity_score = args["space_coefficient"] * num_inliers + (1-args["space_coefficient"]) * len(matches)
    similar_images[img_path] = similarity_score
    print(similarity_score)



threshold = 800

# 根据相似度调整分数
for img_path, similarity_score in similar_images.items():
    if image_path != unliked_image_path:  # 只有当相似度超过阈值时才调整分数
        new_score = scores[matched_images.index(img_path)] + ((similarity_score-threshold) / self_sim_score) * scores[rank_ID[best_index]]
        # new_score = (scores[rank_ID[best_index]] + scores[matched_images.index(img_path)]) / 2
        print(img_path, "old score =", scores[matched_images.index(img_path)], "new score =", new_score)
        scores[matched_images.index(img_path)] = new_score
    
    

# 重新排序分数
rank_ID = np.argsort(-np.array(scores))

# 可视化调整后的结果
figure()
gray()
subplot(5, 4, 1)
imshow(im[:, :, ::-1])
axis('off')

for i in range(min(16, len(rank_ID))):
    print(str(matched_images[rank_ID[i]]))
    img = Image.open(matched_images[rank_ID[i]])
    gray()
    subplot(5, 4, i + 5)
    imshow(img)
    axis('off')

show()
