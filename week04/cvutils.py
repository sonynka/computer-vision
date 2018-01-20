import os
import tensorflow.contrib.keras as k
from shutil import copyfileobj
from sklearn.datasets.base import get_data_home, Bunch
from sklearn.datasets import fetch_mldata
from urllib.request import urlopen

def fetch_mnist(data_home=None):
    # where to store the data
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)        
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    
    # download if needed
    if not os.path.exists(mnist_save_path):
        print("Download MNIST to",mnist_save_path)
        mnist_url = urlopen("http://home.htw-berlin.de/~hezel/files/data/mnist-original.mat")
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)
    return fetch_mldata('MNIST original')

def fetch_cifar10():
    cache_dir = os.path.expanduser(os.path.join('~', '.keras/datasets'))
    print("Download MNIST to", cache_dir)
    (x_train, y_train), (x_test, y_test) = k.datasets.cifar10.load_data()
    train = Bunch(data=x_train, target=y_train)
    test = Bunch(data=x_test, target=y_test)
    return Bunch(train=train, test=test)