# -*- coding:utf-8 -*-

import os
import numpy as np


class Reader:

    # PREFIX = "/Users/app/Desktop/nn_code/result02"
    PREFIX = "/Volumes/ywy/peng/tcflow019"
    POSTFIX = ".dat"
    DATA_SIZE = 4000
    WINDOWS = 200
    ROW_NUMBER = 20

    def __init__(self, files, model):

        self.__domain = []
        self.__codomain = []
        self.__directories = None
        self.__files = files
        self.__label = None
        self.__paths = []
        self.__model = model

        self.__reader = True
        self.__data_path = []

    def __reader_p(self, model):

        content = []
        path = Reader.PREFIX + "/"
        for directory in self.__directories:
            for filename in self.__files:

                if not os.path.isfile(path + str(directory) + "/p" + str(filename) + Reader.POSTFIX):
                    continue

                print (str(directory) + "-" + str(filename) + " Start... ")
                
                start = None
                reader = open(path + str(directory) + "/p" + str(filename) + Reader.POSTFIX, "r").readlines()
                for index in range(len(reader)):
                    if reader[index][0] == "(":
                        start = index + 1
                        break
                for index in range(Reader.DATA_SIZE):
                    if model:
                        content.append([float(reader[start + index].strip())])
                    else:
                        self.__domain.append([float(reader[start + index].strip())])

                if directory not in self.__paths:
                    self.__paths.append(directory)

                assert ((model and len(content) % Reader.DATA_SIZE == 0) or
                        (not model and len(self.__domain) % Reader.DATA_SIZE == 0))

                if str(directory) + "-" + str(filename) not in self.__data_path:
                    self.__data_path.append(str(directory) + "-" + str(filename))
                print (str(directory) + "-" + str(filename) + " Done!")

        return content

    def __reader_u(self, model):

        content = []
        path = Reader.PREFIX + "/"
        for directory in self.__directories:
            for filename in self.__files:

                if not os.path.isfile(path + str(directory) + "/U" + str(filename) + Reader.POSTFIX):
                    continue

                print (str(directory) + "-" + str(filename) + " Start... ")

                start = None
                reader = open(path + str(directory) + "/U" + str(filename) + Reader.POSTFIX, "r").readlines()
                for index in range(len(reader)):
                    if reader[index][0] == "(":
                        start = index + 1
                        break
                for index in range(Reader.DATA_SIZE):
                    values = reader[start + index].split()
                    if model:
                        #content.append([float(values[0][1:]), float(values[1])])
                        content.append([pow(pow(float(values[0][1:]), 2)+pow(float(values[1]), 2), 0.5)])
                    else:
                        #self.__domain.append([float(values[0][1:]), float(values[1])])
                        self.__domain.append([pow(pow(float(values[0][1:]), 2) + pow(float(values[1]), 2), 0.5)])

                if directory not in self.__paths:
                    self.__paths.append(directory)

                assert ((model and len(content) % Reader.DATA_SIZE == 0) or
                        (not model and len(self.__domain) % Reader.DATA_SIZE == 0))

                if str(directory) + "-" + str(filename) not in self.__data_path:
                    self.__data_path.append(str(directory) + "-" + str(filename))
                print (str(directory) + "-" + str(filename) + " Done !")
        return content

    def get_train_data(self, window_size):

        self.__reader = False

        assert (Reader.WINDOWS % window_size == 0)

        # print("Train: ", self.__domain[0], len(self.__domain))

        self.__domain = np.reshape(self.__domain,
                                   (len(self.__codomain),
                                    Reader.ROW_NUMBER,
                                    Reader.WINDOWS,
                                    len(self.__domain[0])))
        train_data = []
        train_label = []
        train_data_path = []
        for index in range(0, Reader.WINDOWS, window_size):
            train_data += self.__domain[:, :, index:index+window_size, :].tolist()

        for item in self.__codomain:
            for index in range(int(Reader.WINDOWS/window_size)):
                train_label.append(item)

        for item in self.__data_path:
            for index in range(int(Reader.WINDOWS/window_size)):
                train_data_path.append(item)

        # print(len(train_data), len(train_label))
        return train_data, train_label, train_data_path

    def get_test_data(self, window_size):

        self.__reader = False

        # print("Test: ", self.__domain[0], len(self.__domain))

        assert (Reader.WINDOWS % window_size == 0 and len(self.__files) == 1)

        self.__domain = np.reshape(self.__domain,
                                   (len(self.__codomain),
                                    Reader.ROW_NUMBER,
                                    Reader.WINDOWS,
                                    len(self.__domain[0])))

        return self.__domain[:, :, 0:window_size, :], self.__codomain

    def reader(self, directories, label):

        assert (self.__reader == 1)

        self.__directories = directories
        self.__label = label

        if self.__model == 1:
            self.__reader_p(False)
        elif self.__model == 2:
            self.__reader_u(False)
        elif self.__model == 3:
            content_t = self.__reader_p(True)
            content_u = self.__reader_u(True)
            for index in range(len(content_t)):
                self.__domain.append(content_t[index] + content_u[index])
        for index in range(len(self.__codomain)*Reader.DATA_SIZE, len(self.__domain), Reader.DATA_SIZE):
            self.__codomain.append([self.__label])

    @staticmethod
    def create_directories(root, start, end):

        directories = []
        for parent, dir_names, file_names in os.walk(root):
            for dir_name in dir_names:
                try:
                    if start <= float(dir_name[2:]) <= end:
                        if len(directories) == 0:
                            directories.append(dir_name)
                        else:
                            signal = True
                            for index in range(len(directories)):
                                if float(dir_name[2:]) < float(directories[index][2:]):
                                    directories.insert(index, dir_name)
                                    signal = False
                                    break
                            if signal:
                                directories.append(dir_name)
                except Exception:
                    pass
        return directories


if __name__ == "__main__":

    reader = Reader(range(1990, 2000, 10), 3)
    reader.reader(Reader.create_directories(95.6, 97.2, 1.6), [1])
    reader.reader(Reader.create_directories(81.6, 100, 1.6), [0])
    print(reader.get_train_data(Reader.WINDOWS)[0])

'''
    for window_size in range(10, 200, 10):
        train_domain, train_codomain = reader.get_train_data(window_size)
        test_domain, test_codomain, test_files = reader.get_test_data(window_size)
'''



