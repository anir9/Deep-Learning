import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from scipy import stats
#import statistics



# load mini training data and labels
mini_train = np.load('knn_minitrain.npy')
mini_train_label = np.load('knn_minitrain_label.npy')
# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10,2)

#print(mini_train)
#print(mini_train_label)

# Define knn classifier
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
        for label, train_point in zip(labels,dataSet):   # [(1,(2,3))]
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

outputlabels=kNNClassify(mini_test,mini_train,mini_train_label,4)

print ('random test points are:', mini_test)
print ('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:,0]
train_y = mini_train[:,1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label==0)], train_y[np.where(mini_train_label==0)], color='red')
plt.scatter(train_x[np.where(mini_train_label==1)], train_y[np.where(mini_train_label==1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label==2)], train_y[np.where(mini_train_label==2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label==3)], train_y[np.where(mini_train_label==3)], color='black')

test_x = mini_test[:,0]
test_y = mini_test[:,1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels==0)], test_y[np.where(outputlabels==0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels==1)], test_y[np.where(outputlabels==1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels==2)], test_y[np.where(outputlabels==2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels==3)], test_y[np.where(outputlabels==3)], marker='^', color='black')

#save diagram as png file
plt.savefig("miniknn.png")