import numpy as np
import math
import sys

def mnist_data_load():
    '''
    IDX file format is a binary file format.
    Each byte from 16th byte onward, contains pixel data, and the type of the data is unsigned byte, i.e. each pixel data is contained in a 8-bit binary digit, 
    having value from 0 to 255.
    the value of the pixel at the (0,0) 
    Similarly, the 18th, 19th and 20th byte in the file will give us the values of the pixels at the co-ordinates (0,2), (0,3), (0,4) respectively.
    '''
    train_images_file=open('train-images.idx3-ubyte','rb')
    train_label_file=open('train-labels.idx1-ubyte','rb')
    test_images_file=open('t10k-images.idx3-ubyte','rb')
    test_label_file=open('t10k-labels.idx1-ubyte','rb')

    '''
    If byteorder is "big", the most significant byte is at the beginning of the byte array. 
    Return the integer represented by the given array of bytes.
    '''
    train_images_file.read(16)
    train_label_file.read(8)
    train_x=np.zeros((60000,28*28),dtype='uint8')
    train_y=np.zeros(60000,dtype='uint8')
    
    test_images_file.read(16)
    test_label_file.read(8)
    test_x = np.zeros((10000, 28*28),dtype='uint8')
    test_y =np.zeros(10000,dtype='uint8')

    for i in range(60000):
        for j in range(28*28):
            train_x[i,j]=int.from_bytes(train_images_file.read(1),byteorder='big')
        train_y[i]=int.from_bytes(train_label_file.read(1),byteorder='big')

    for i in range(10000):
        for j in range(28*28):
            test_x[i,j] = int.from_bytes(test_images_file.read(1), byteorder='big')
        test_y[i]=int.from_bytes(test_label_file.read(1),byteorder='big')

    return (train_x,train_y),(test_x,test_y)

def prior(train_y):
    '''
    prior is the label 
    prior = sum of each label category/sum of label data
    prior(label=0) = 5923 data/60000 data
    and so on for each label (0-9)
    '''
    prior=np.zeros(10)
    for y in range(10):
        prior[y]=np.sum(train_y==y)/len(train_y)
    return prior

def print_MNIST_imagination(pixvalueProb,mode,data_len,err_rate):
    if mode =='0': thresh=2
    elif mode =='1':thresh=128
    print('Imagination of numbers in Bayesian classifier:\n')
    for label in range(10):
        print('{}:'.format(label))
        for i in range(28):
            for j in range(28):
                print('1' if np.argmax(pixvalueProb[label,i*28+j])>=thresh else '0',end=' ')
            print()
        print()
    print()
    print('Error rate: {:.4f}'.format(err_rate / data_len))
    print()

def pix_prob_discrete(train_x,train_y):
    '''  
    Tally the frequency of the values of each pixel into 32 bins. 
    get pix value to conditional prob on pixel class&dim
    10 matrix with 784 row and 32 column(bins)
    train x have 60000 array row with 28*28 column (representation of each pixel)
    8 is 256/32
    return: (#label,#dim(rows),#column bins(value)):(10,784,32) ndarray
    '''
    print('Processing the pixel probability distribution into 32 bins for each class')
    pix_prob_distribution=np.zeros((10,28*28,32))
    for i in range(len(train_x)):
        label=train_y[i]
        if i % 6000 == 0:
                print("-> Processing %d images" % i)
        if i == 59999:
            print("Processing 60000 training images is Done!")
        for row in range(28*28):
            pix_prob_distribution[label][row][int(train_x[i,row])//8]+=1

    #to get the probability distribution for each pixel, devide by the total value
    for label in range(10):
        for row in range(28*28):
            count=0
            for bins in range(32):
                count+=pix_prob_distribution[label][row][bins]
            pix_prob_distribution[label][row][:]/=count
    return pix_prob_distribution

def discrete_mode(data_len,pixProbDist,prior,test_x,test_y):
    err_rate=0
    '''
     Calculate the log likelihood of data given the class
     Calculate log of pixel prob dist data and avoid log(0) = (NaN)value in data with given small number (1e-6)
     Calculate the log of prior
     Calculate in log form so we use addition to add likelihood and prior
    '''
    print("Tally the posterior of testing images")
    for i in range(data_len):
        posterior = np.zeros(10)
        if i % 9999: print("Testing image %i is Done!" %i)
        for label in range(10):
            for row in range(28 * 28):  #Naive bayes classifier in log form
                    posterior[label] += np.log(max(1e-6,pixProbDist[label, row, int(test_x[i, row])//8]))
            posterior[label] += np.log(prior[label]) #tally the log of prior(label)
        
        '''
        the logs would all be between −∞ and zero, as would their sum.
        normalization make the positive value,so argmax becomes argmin.
        '''
        posterior /= np.sum(posterior) # normalized
        print('Posterior (in log scale):')
        for label in range(10):
            print('{}: {}'.format(label, posterior[label]))
        prediction=np.argmin(posterior)
        print('Prediction: {}, Ans: {}'.format(prediction, test_y[i]))
        print()
        if prediction != test_y[i]: err_rate += 1
    return err_rate


def pix_prob_continuous(train_x,train_y):
    '''
    get the mean , variance and gasussian distribution for each pixel with range (0 to 255)
    :return: (#class,#dim,#pix_value):(10,784,256) ndarray
    '''
    pix_prob_distribution =np.zeros((10,28*28,256))
    for label in range(10):  # 10 classes        
        #select the all pixel for each class
        data_train_pixel = train_x[train_y == label]
        for i in range(28*28):
            mean = np.mean(data_train_pixel[:, i])
        
            var = np.var(data_train_pixel[:, i])
            if var != 0: var
            else: var = 0.1
                
    #         print(var)
            for j in range(256):
                pix_prob_distribution[label,i,j] =(1/math.sqrt(2*math.pi*var))*math.exp((-(j-mean)**2)/(2*var))
    return pix_prob_distribution

def test_continuous(data_len,pixProbDist,prior,test_x,test_y):
    err_rate = 0
    for i in range(data_len):
        if i % 9999: print("Testing image %i is Done!" %i)
        posterior = np.zeros(10)
        for label in range(10):
            for d in range(28 * 28):  
                posterior[label] += np.log(max(1e-6,pixProbDist[label, d, int(test_x[i, d])]))
            posterior[label] += np.log(prior[label])
        
        posterior /= np.sum(posterior)
        print('Posterior (in log scale):')
        for label in range(10):
            print('{}: {}'.format(label, posterior[label]))
        prediction=np.argmin(posterior)
        print('Prediction: {}, Ans: {}'.format(prediction, test_y[i]))
        print()
        if prediction != test_y[i]: err_rate += 1
    return err_rate

if __name__=='__main__':
    print('Load the MNIST data, wait a moment...')
    (train_x,train_y),(test_x,test_y)=mnist_data_load()
    
    mode=input('Mode (0:discrete / 1:continuous): ')
    if mode=='0':
        prior=prior(train_y)
        pixProbDist = pix_prob_discrete(train_x, train_y)
        result = discrete_mode(len(test_y), pixProbDist,prior, test_x, test_y)
        print_MNIST_imagination(pixProbDist,mode,len(test_y),result)

    elif mode=='1':
        prior=prior(train_y)
        pixProbDist=pix_prob_continuous(train_x,train_y)
        result = test_continuous(len(test_y),pixProbDist,prior,test_x,test_y)
        print_MNIST_imagination(pixProbDist,mode,len(test_y),result)
    else:
        raise Exception("Error, Only have 2 modes: 0 for discrete or 1 for continuous")