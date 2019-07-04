from PIL import Image 
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sn
 
 
dic_number = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'zero':0}

def img2vector(filename):
    returnVect = np.zeros((1, 400))
    img = Image.open(filename)
    returnVect = np.array(img).reshape(1,400)
    return returnVect
 

def read_test_train(filename):
    hwLabels = []
    FileList = listdir(filename)
    m = len(FileList)
    Mat = np.zeros((m, 400))
    for i in range(m):
        fileNameStr = FileList[i]
        classNumber = int(dic_number[fileNameStr.split('_')[0]])
        hwLabels.append(classNumber)
        Mat[i,:] = img2vector('./data/%s' % (fileNameStr))

    return Mat, np.array(hwLabels)

if __name__=='__main__':
    X, y = read_test_train('./data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)

    neigh =KNN(n_neighbors = 5, algorithm = 'auto')
    neigh.fit(X_train, y_train)
    
    errorCount = 0
    y_pre = []
    mTest = len(y_test)
    for i in range(mTest):
        classNumber = y_test[i]
        vectorUnderTest = X_test[i,:]
        #获得预测结果
        vectorUnderTest = vectorUnderTest.reshape(1,-1)
        classifierResult = neigh.predict(vectorUnderTest)
        y_pre.append(classifierResult)
        # print("The result of KNN is %d\tThe true result is %d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    y_pre = np.array(y_pre)
    confusion_m = confusion_matrix(y_test, y_pre)

    print("The count of the wrong results is %d\nThe error rate is %.3f%%" % (errorCount, errorCount/mTest * 100))
    print("The confusion matrix is:")
    print(confusion_m)
    print("Per-class precision:")
    for i in range(10):
        rate = precision_score(y_test, y_pre, labels=[i], average='macro')
        print("Number %d :%.3f%%" % (i,rate))
    print("The recall is %.3f%%" % recall_score(y_test, y_pre, average='micro'))
