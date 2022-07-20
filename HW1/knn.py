import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
def kNNClassify(newInput, dataSet, labels, k): 
    result=[]
    ########################
    # Input your code here #
    ########################
    distances=[]
    
    def l2(x, y):
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
        test_point = test_point.flatten()
        for label, train_point in zip(labels,dataSet):   # [(1,(2,3))]
            train_point = train_point.flatten()
            euclidean_distance = l2(test_point, train_point)
            distances.append([euclidean_distance, label])
        
        #print(distances, "old")
        distances = np.array(distances)
        sorted_dist = distances[np.argsort(distances[:,0])]
        #print(distances, "new")
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

start_time = time.time()
outputlabels=kNNClassify(x_test[0:20],x_train,y_train,10)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
