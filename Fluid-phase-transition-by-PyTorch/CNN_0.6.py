import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as fun
import numpy as np
from torchvision import transforms
import os

torch.manual_seed(1)
torch.cuda.manual_seed(1)

EPOCH = 4
BATCH_SIZE = 50
LR = 0.001


WIDTH = 200
HEIGHT = 20

history = {
    "loss": [],
    "acc": []
}


def create_directories(root, start, end):

    directories = []
    for parent, dir_names, file_names in os.walk(root):
        for dir_name in dir_names:
            try:
                value = float(dir_name[2:])
                if start <= value <= end:
                    place = len(directories)
                    for index in range(len(directories)):
                        if value < float(directories[index][2:]):
                            place = index
                            break
                    directories.insert(place, dir_name)
            except Exception:
                pass
    print(len(directories))
    return directories


def default_loader(image_path):

    p_path, u_path = image_path
    p_file = open(p_path, "r").readlines()
    for p_index in range(len(p_file)):
        if p_file[p_index][0] == "(":
            p_start = p_index + 1
            break
    p_data = []
    for index in range(WIDTH*HEIGHT):
        p_data.append(float(p_file[p_start + index].strip()))

    data = []
    u_file = open(u_path, "r").readlines()
    for u_index in range(len(u_file)):
        if u_file[u_index][0] == "(":
            u_start = u_index + 1
            break
    for index in range(WIDTH*HEIGHT):
        values = u_file[u_start + index].split()
        data.append([p_data[index], pow(pow(float(values[0][1:]), 2)+pow(float(values[1]), 2), 0.5)])
    data = np.reshape(data, (HEIGHT, WIDTH, 2))
    # if save:
    #    plt.imsave(t_path[:-4]+".jpg", data)
    #    print(t_path[:-4]+".jpg, save Done !")
    return data


class MyDataSet(Dataset):

    def __init__(self, root, parameter, transform=None,
                 target_transform=None,
                 loader=default_loader):
        self.__images = []
        for label in [1, 0]:
            for dirs in parameter[label][0]:
                for file_name in parameter[label][1]:
                    p_path = os.path.join(root, dirs + "/p" + str(file_name) + ".dat")
                    u_path = os.path.join(root, dirs + "/U" + str(file_name) + ".dat")
                    if os.path.isfile(p_path) and os.path.isfile(u_path):
                        self.__images.append(((p_path, u_path), label))
        self.__transform = transform
        self.__target_transform = target_transform
        self.__loader = loader

    def __getitem__(self, index):

        image_path, image_label = self.__images[index]
        image = self.__loader(image_path)
        if self.__transform is not None:
            image = self.__transform(image)
        return image, image_label

    def __len__(self):
        return len(self.__images)


TRAIN_ROOT = "D:/peng/tcflow0717"
TEST_ROOT = "D:/peng/tcflow0717"

TRAIN_PARAMETER = {1: (create_directories(TRAIN_ROOT, 10.8, 68.16), range(1800, 2010, 20)),
                   0: (create_directories(TRAIN_ROOT, 73.68, 120.0), range(1800, 2010, 20))}


TEST_PARAMETER = {1: (create_directories(TEST_ROOT, 66.0, 71.52), [2000]),
                  0: (create_directories(TEST_ROOT, 71.76, 91.2), [2000])}


train_data = MyDataSet(TRAIN_ROOT, TRAIN_PARAMETER, transform=transforms.ToTensor())
test_data = MyDataSet(TEST_ROOT, TEST_PARAMETER, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

CLASS_NUMBER = 2


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.__convolution_1 = nn.Sequential(
            nn.Conv2d(in_channels=2,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.__convolution_2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.__output_layer = nn.Sequential(
            nn.Linear(32*5*50, 2),
        )

    def forward(self, x_input):
        x_input = self.__convolution_1(x_input)
        x_input = self.__convolution_2(x_input)
        x_input = x_input.view(x_input.size(0), -1)
        output_result = self.__output_layer(x_input)
        return fun.softmax(output_result, dim=1)


def train():

    net = AlexNet()
    net.double()

    is_gpu = torch.cuda.is_available()

    print("is gpu: ", is_gpu)

    if is_gpu:
        device = torch.device("cuda:0")
        net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        loss_value = 0.
        acc_value = 0.
        iter_count = 0
        for batch_x, batch_y in train_loader:
            if is_gpu:
                batch_y = batch_y.type(torch.cuda.LongTensor)
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = net(batch_x)
            loss = loss_fun(out, batch_y)
            loss_value += loss.data.item()
            predict = torch.max(out, 1)[1]
            correct = (predict == batch_y).sum()
            acc_value += correct.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_count += BATCH_SIZE
            if iter_count % 100 == 0:
                print("After %d iter_count ==> Train Loss: %f, Acc: %f"
                      % (iter_count, loss_value / iter_count, acc_value / iter_count))
                history["loss"].append(loss_value / iter_count)
                history["acc"].append(acc_value / iter_count)
        print("After %d epoch ==> Train Loss: %f, Acc: %f"
              % (epoch, loss_value/len(train_data), acc_value/len(train_data)))
        validate(model=net, epoch=epoch)

    torch.save(net, '0717_model.pkl')

    output = open("0717_loss_history.txt", "w")
    for index in range(len(history["loss"])):
        output.write(str(history["loss"][index]) + " ")
    output.close()

    output = open("0717_acc_history.txt", "w")
    for index in range(len(history["acc"])):
        output.write(str(history["acc"][index]) + " ")
    output.close()


def validate(model, epoch, result=False):

    is_gpu = torch.cuda.is_available()

    truth_value = []
    predict_value = []

    acc_value = 0.
    for batch_x, batch_y in test_loader:
        if result:
            truth_value += batch_y.data.numpy().tolist()
        if is_gpu:
            batch_y = batch_y.type(torch.cuda.LongTensor)
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        predict = torch.max(out, 1)[1]
        correct = (predict == batch_y).sum()
        acc_value += correct.data.item()

        if result:
            p_value = out.cpu().data.numpy().tolist()
            for item in p_value:
                predict_value.append(item[1])

    print("After %d epoch ==> Test Acc: %f" % (epoch, acc_value / len(test_data)))

    return truth_value, predict_value


def show(truth_value, predict_value):

    x = []
    for index in range(len(truth_value)):
        x.append(index)
    dir_name_list = []
    for label in [1, 0]:
        for dir_name in TEST_PARAMETER[label][0]:
            if os.path.isdir(os.path.join(TEST_ROOT, dir_name)):
                dir_name_list.append(dir_name)

    out = open("0717_predict.txt", "w")
    for index in range(len(predict_value)-1):
        out.write(str(predict_value[index])+","+dir_name_list[index]+"),(")
    out.write(str(predict_value[-1]) + "," + dir_name_list[-1])
    out.close()

    plt.plot(x, truth_value, "g*")
    plt.plot(x, predict_value, "r+")

    plt.plot(x, truth_value, "g")
    plt.plot(x, predict_value, "r")

    print("truth turn point: ", dir_name_list[index])

    for index in range(len(dir_name_list)):
        if index % 5 == 0:
            continue
        dir_name_list[index] = ""

    plt.xticks(x, dir_name_list, rotation=90)

    plt.show()


def load_model(model_path):

    return torch.load(model_path)


if __name__ == "__main__":

    # train()
    le_net = load_model("0717_model.pkl")
    t, p = validate(model=le_net, epoch=EPOCH, result=True)
    show(t, p)















