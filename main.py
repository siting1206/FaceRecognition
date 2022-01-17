import numpy as np
import os, glob, random, cv2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings
warnings.filterwarnings('ignore')

def loadImageSet(path='./FaceData', Count=5):  #read pgm files(choose 5 pieces randomly in each folder)
    X_Train = []  #training data
    X_Test = []   #testing data
    y_Train = []  #training label
    y_Test = []   #testing label
    for k in range(1, 41):
        folder = os.path.join(path, 's%d' % (k))
        data = [cv2.imread(d, 0) for d in glob.glob(os.path.join(folder, '*.pgm'))]
        sample = random.sample(range(10), Count)
        X_Train.extend([data[i].ravel() for i in range(10) if i in sample])
        X_Test.extend([data[i].ravel() for i in range(10) if i not in sample])
        y_Train.extend([k] * Count)
        y_Test.extend([k] * (10 - Count))
    return np.array(X_Train), np.array(y_Train), np.array(X_Test), np.array(y_Test)


def pca(dataMat, n):
    meanVal = np.mean(dataMat, axis=0)#compute mean values
    newData = dataMat - meanVal
    covMat = np.cov(newData, rowvar=0) #compute correlation matrix, rowvar=0: each line(row) of data is a sample
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))#find eigenvectors and eigenvalues
    eigValInd = np.argsort(eigVals)#sorted from low to high
    eigValInd = eigValInd[:-(n+1):-1]#choose n features from the back of the sorted data
    redEigVects = eigVects[:,eigValInd]#corresponding to vectors
    lowDDataMat = newData * redEigVects#transform the data set
    reconMat = (lowDDataMat * redEigVects.T) + meanVal#reconstruct the data set
    return reconMat

def LDA_model(xTrain, xTest, yTrain):
    lda = LDA()
    xTrain = lda.fit_transform(xTrain, yTrain)
    xTest = lda.transform(xTest)
    return xTrain, xTest

#LDA defined by myself
'''def lda(dataMat, target, n):
    clusters = np.unique(target)
    
    #compute within_class scatter matrix
    Sw = np.zeros((dataMat.shape[1], dataMat.shape[1]))
    for i in clusters:
        datai = dataMat[target == i]
        datai = datai-datai.mean(0)
        Swi = np.mat(datai).T*np.mat(datai)
        Sw += Swi
        
    #compute between_class scatter matrix
    SB = np.zeros((dataMat.shape[1], dataMat.shape[1]))
    u = dataMat.mean(0)
    for i in clusters:
        Ni = dataMat[target == i].shape[0]
        ui = dataMat[target == i].mean(0)
        SBi = Ni*np.mat(ui-u).T*np.mat(ui-u)
        SB += SBi
    S = np.linalg.inv(Sw).dot(SB)
    eigVals, eigVects = np.linalg.eig(S) #find eigenvectors and eigenvalues
    eigValInd = np.argsort(eigVals) #sorted data from low to high
    eigValInd = eigValInd[:(-n-1):-1] #choose n features from the back of the sorted data
    w = eigVects[:,eigValInd] #corresponding to vectors
    lowDDataMat = np.dot(dataMat, w) #transform the data set
    return  lowDDataMat'''

def plot_confusion_matrix(name, dimension, confusion_mat): #confusion matrix
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
    plt.title('{} Dimension {} Confusion matrix'.format(name, dimension))
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def classifier(dimension, xTrain, xTest, yTrain, yTest, tech): #SVM Classifier
    svm = SVC(kernel='linear')
    svm.fit(np.array(xTrain), np.float32(yTrain))
    yPredict = svm.predict(np.float32(xTest))
    if(tech == "pca"):
        print('%d維的辨識率(PCA): %.2f%%' % (dimension, (yPredict == np.array(yTest)).mean() * 100))
    else:
        print('%d維的辨識率(LDA): %.2f%%' % (dimension, (yPredict == np.array(yTest)).mean() * 100))
    return yTest.tolist(), yPredict

def main(dimension, xTrain_, yTrain, xTest_, yTest):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(xTrain_[:, 0].flatten(), xTrain_[:, 1].flatten(), marker='^', s=90) #original data set
    #pca transform
    xTrain = pca(xTrain_, dimension)
    xTrain = np.asarray(xTrain).astype(float)
    #ax.scatter(xTrain[:, 0].flatten(), xTrain[:, 1].flatten(), marker='o', s=10, c='red') #transformd data set by pca
    #plt.show()
    xTest = pca(xTest_, dimension)
    xTest = np.asarray(xTest).astype(float)
    result, ans = classifier(dimension, xTrain, xTest, yTrain, yTest, "pca")
    confusion_mat = confusion_matrix(ans, result)
    plot_confusion_matrix("PCA", dimension, confusion_mat)

    # LDA transform
    xTrain, xTest = LDA_model(xTrain, xTest, yTrain)
    result, ans = classifier(dimension, xTrain, xTest, yTrain, yTest, "lda")
    confusion_mat = confusion_matrix(ans, result)
    plot_confusion_matrix("PCA & LDA", dimension, confusion_mat)

if __name__ == '__main__':
    xTrain_, yTrain, xTest_, yTest = loadImageSet()
    main(10, xTrain_, yTrain, xTest_, yTest)
    main(20, xTrain_, yTrain, xTest_, yTest)
    main(30, xTrain_, yTrain, xTest_, yTest)
    main(40, xTrain_, yTrain, xTest_, yTest)
    main(50, xTrain_, yTrain, xTest_, yTest)