import os
from reader import Reader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml

# save root
ROOT = "/Volumes/ywy/TC_0.6"


def get_data(model, data):

    x = np.zeros(Reader.DATA_SIZE)
    y = np.zeros(Reader.DATA_SIZE)
    z = np.zeros(Reader.DATA_SIZE)

    windows = Reader.WINDOWS / 2
    rows = Reader.ROW_NUMBER * 2

    for index in range(Reader.DATA_SIZE):
        x[index] = index % windows
        y[index] = index / windows
        if model:
            z[index] = data[index][0]
        else:
            z[index] = pow(pow(data[index][1], 2) + pow(data[index][2], 2), 0.5)

    xi = np.linspace(1, windows, windows)
    yi = np.linspace(1, rows, rows)
    zi = ml.griddata(x, y, z, xi, yi, interp="linear")
    return x, y, xi, yi, zi


def save_figure(path, data, filename):

    #plt.contour(data[2], data[3], data[4], 0, linewidths=0, colors='k')
    #plt.pcolormesh(data[2], data[3], data[4], cmap=plt.get_cmap('rainbow')) # cmap=plt.get_cmap('jet')
    plt.contour(data[2], data[3], data[4], 15, linewidths=0.1, colors='k')
    plt.pcolormesh(data[2], data[3], data[4], cmap=plt.get_cmap('jet'))

    plt.colorbar()
    plt.scatter(data[0], data[1], marker='o', c='b', s=0, zorder=10)
    plt.xlim(1, Reader.WINDOWS/2)
    plt.ylim(1, Reader.ROW_NUMBER*2)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "/" + filename + ".jpg")
    plt.close()


def computer():

    global ROOT

    for direction in Reader.create_directories(Reader.PREFIX, 1.6, 190.4):
        for file in range(2000,2010 , 10):
            reader = Reader([file], 3)
            reader.reader([direction], [1])
            data = reader.get_train_data(Reader.WINDOWS)[0]
            data = np.reshape(data, (Reader.DATA_SIZE, 3))
            path = ROOT + "/" + str(direction)
            if len(data) > 0:
                save_figure(path, get_data(True, data), "T" + str(file))
                save_figure(path, get_data(False, data), "U" + str(file))
                print("%s ==> Save Done !" % (path+"/"+str(file)+".jpg"))
            else:
                print("%s ==> No such file ! Skip Done !" % (path+"/"+str(file)+".jpg"))
            #input(">>> ")
    print("Working Done !")


if __name__ == "__main__":

    computer()








