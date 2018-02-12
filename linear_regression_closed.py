import numpy as np
import matplotlib.pyplot as plt

# wrote in 2016

def f(x):
    return x+x*2*np.sin(x)


class regression:
    def __init__(self,train_x,train_y,test_x,test_y):
        self.test_x = test_x
        self.train_x = train_x
        self.train_y = train_y
        self.test_y = test_y

    def linearRegression(self):
        xTx = train_x.T.dot(train_x)
        if np.linalg.det(xTx) ==0.0:
            print("singular")
            return
        w = np.linalg.inv(xTx).dot(self.train_x.T).dot(self.train_y)
        pre_y = self.test_x.dot(w)
        return pre_y

    def bias_variance(self,pre_y):
        var = np.mean((pre_y - np.mean(pre_y)) ** 2)
        bias2 = np.mean(np.mean(pre_y) - self.test_y) ** 2
        noise = np.mean((pre_y - self.test_y) ** 2)
        print("var = ", var)
        print("bias2 = ", bias2)
        print("noise = ", noise)
    def plotAns(self,pre_y):
        plt.plot(self.test_y, 'ro', label='Original data')
        plt.plot(pre_y, label='Fitted line')
        plt.show()

if __name__=="__main__":
    train_x = (np.arange(10000)/100)
    train_y = f(train_x)
    test_x = train_x + np.random.random()
    train_x = np.column_stack((train_x,np.ones(10000)))

    test_x = np.column_stack((test_x, np.ones(10000)))
    test_y = f(test_x[:,0])


    regressor = regression(train_x,train_y,test_x,test_y)
    pre_y = regressor.linearRegression()
    regressor.bias_variance(pre_y)
    regression(train_x, train_y, test_x, test_y).LWLR(0.1)
    regressor.plotAns(pre_y)