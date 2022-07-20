import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mini_train = np.array([[0,1,0], [0,1,1], [1,2,1], [1,2,0], [1,2,2], [2,2,2], [1,2,-1], [2,2,3], [-1,-1,-1], [0,-1,-2], [0,-1,1], [-1,-2,1]])

mini_train_label = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"])

mini_test = np.array([1,0,1])
mini_test = mini_test.reshape(1,3)

# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
    result=[]
    ########################
    # Input your code here #
    ########################
    distances=[]
    
    def l2(x, y):
        #print(x,y)
        d = np.sqrt(np.sum(np.square(x-y)))
        return d

    def mode(arr):
        nums = dict()
        for i in arr:
            if i not in nums:
                nums[i] = 1
            else:
                nums[i] += 1
        max_val = max(nums, key=nums.get)
        return max_val 

    for test_point in newInput:  # for every data pt in the test set
        for label, train_point in zip(labels,dataSet):   # [(1,(2,3))]
            euclidean_distance = l2(test_point, train_point)
            distances.append([euclidean_distance, label])
        
        #print(distances, "old")
        distances = np.array(distances)
        sorted_dist = distances[np.argsort(distances[:,0])]
        #print(sorted_dist, "new")
        knn_list = sorted_dist[:k] # list of KNNs w/ labels and test pt of reference
        #print(knn_list)
        new_list = []
        for i,j in knn_list:
            new_list.append(j)
            

        new_label = mode(new_list)
        #print(type(new_label))
        result.append(new_label) # save label to test data pt

        distances = [] # ready to test next point
    
    ####################
    # End of your code #
    ####################
    return result

outputlabels1=kNNClassify(mini_test,mini_train,mini_train_label,1)
outputlabels2=kNNClassify(mini_test,mini_train,mini_train_label,2)
outputlabels3=kNNClassify(mini_test,mini_train,mini_train_label,3)

print ('random test points are:', mini_test)
print ('knn classfied labels for test when k = 1:', outputlabels1)
print ('knn classfied labels for test when k = 2:', outputlabels2)
print ('knn classfied labels for test when k = 3:', outputlabels3)