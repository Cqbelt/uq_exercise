import numpy as np
import pickle
import matplotlib.pyplot as plt
import math

# 定义模型
class softmax_classifer:
    def __init__(self):
        self.W = None

    def loss(self,X,y,reg):
        """
        loss function：不同于svm定义的损失函数，
        """
        loss = 0.0
        dW = np.zeros_like(self.W)    # D by C
        num_train, dim = X.shape

        f = X.dot(self.W)    # N by C
        # Considering the Numeric Stability
        f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1
        prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
        y_trueClass = np.zeros_like(prob)
        y_trueClass[range(num_train), y] = 1.0    # N by C

        # 计算损失  y_trueClass是N*C维度  np.log(prob)也是N*C的维度
        loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(self.W * self.W)

        # 计算损失  X.T = (D*N)  y_truclass-prob = (N*C)
        dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * self.W

        return loss, dW
    
    def train(self,X,y,learning_rate=1e-5,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
        num_train,dim = X.shape
        num_class = np.max(y)+1   
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_class)  #权重的初始化
        
        # Run SGD to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            idx_batch = np.random.choice(num_train,batch_size,replace=True)
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
        
            # evaluate loss and gradient
            loss, grad =self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= learning_rate * grad
        
            if verbose and it % 100 == 0:
                print('itenration %d / %d:loss %f' % (it,num_iters,loss))
            
        return loss_history
    
    def predict(self,X):
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = np.argmax(scores,axis=1)
        return y_pred,np.array([scores[i][y_pred[i]] for i in range(len(y_pred))])
        
if __name__ == '__main__':
    # 导入数据
    train_data = []
    train_label = []
    for i in range(1,6):
        file_object = open('data/cifar-10-batches-py/data_batch_'+str(i),'rb')
        data_object = pickle.load(file_object,encoding='bytes')  # 字典格式
        # print(data_object[b'data'])
        for line in data_object[b'data']:
            train_data.append(line)
        for line in data_object[b'labels']:
            train_label.append(line)
    # notice there,train_data and train_label are the structure of python's list,you should transport to numpy's array
    train_data = np.array(train_data).astype("float")
    train_label = np.array(train_label)
    print("train_data shape:"+str(train_data.shape))
    print("train_label shape:"+str(train_label.shape))

    #%%
    test_data = []
    test_label = []
    test_file = open('data/cifar-10-batches-py/test_batch','rb')
    test_file_object = pickle.load(test_file,encoding='bytes')
    # print(test_file_object)
    for line in test_file_object[b'data']:
        test_data.append(line)
    for line in test_file_object[b'labels']:
        test_label.append(line)
    test_data = np.array(test_data).astype("float")
    test_label = np.array(test_label)
    print("test_data shape:"+str(test_data.shape))
    print("test_label shape:"+str(test_label.shape))
    # print(test_label.ndim)

    mean_image = np.mean(train_data,axis=0)
    train_data -=mean_image
    train_data = train_data/0.27421
    # train_label -=mean_image
    test_data -=mean_image
    # test_label -=mean_image

    train_data = np.hstack((train_data,np.ones([train_data.shape[0],1])))
    # train_label = np.hstack(train_label,np.ones([train_label.shape[0],1]))
    test_data = np.hstack((test_data,np.ones([test_data.shape[0],1])))
    # test_label = np.hstack(test_label,np.ones([test_label.shape[0],1]))
    print(str(train_data.shape))        
    softmax = softmax_classifer()

    loss_hist = softmax.train(train_data, train_label, learning_rate=1e-7, reg=2.5e4,
                          num_iters=1500, verbose=True)

    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    y_train_pred = softmax.predict(train_data)
    print("train accuracy: %f" % (np.mean(train_label == y_train_pred)))
    y_test_pred = softmax.predict(test_data)
    print('accuracy: %f' % (np.mean(test_label== y_test_pred)))
