import torch
import pandas
import numpy as np
import torch.optim as optim
import PIL
from PIL import Image, ImageGrab, ImageOps
from tkinter import *
import network as nn

TRAINING = False
MODEL_TO_USE = './models/v2.pt'
net = nn.Net()

canvas = None

INPUT_SIZE = (28, 28)
batch_size = 5


def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y


def draw_line(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y),
                       fill='black',
                       width=13)
    lasx, lasy = event.x, event.y


def clear_canvas(event):
    canvas.delete('all')


def snapshot(event):
    canvas.postscript(file='./drawings/drawing.eps')
    img = PIL.Image.open('./drawings/drawing.eps')
    img.thumbnail(INPUT_SIZE)
    img = ImageOps.grayscale(img)
    img.save('./drawings/drawing.png')

    im_array = np.array(img).astype(np.single)
    im_array = 255 - im_array
    im_array = im_array / 255
    im_array = torch.from_numpy(im_array)

    print(predict(im_array))


def predict(input):
    if isinstance(net,nn.ConNet):
        inputs = input.view(1, 1, 28, 28)
        for i in range(batch_size -1):
            inputs = torch.cat((inputs,input.view(1, 1, 28, 28)))
        predictions = net(inputs)
        predicted = torch.argmax(predictions[0]).item()
    else:
        predictions = net(input.view(1,28*28))
        predicted = torch.argmax(predictions).item()
    return predicted


def main():
    global net
    net = model(MODEL_TO_USE)

    print('Neural Network Ready')

    app = Tk()
    app.geometry("400x400")

    global canvas
    canvas = Canvas(app, bg='white')
    canvas.pack(anchor='nw', fill='both', expand=1)

    canvas.bind("<Button-1>", get_x_and_y)
    canvas.bind("<B1-Motion>", draw_line)

    canvas.bind("<Button-3>", snapshot)
    canvas.bind("<Button-2>", clear_canvas)
    app.mainloop()


def model(save_name):
    global net
    if TRAINING:
        # load in training data
        training = pandas.read_csv('./data/train.csv')

        labels = training.iloc[:, 0].values
        features = training.iloc[:, 1:].values
        train_features = np.array(features).reshape((len(features), 28, 28)).astype(np.single)

        # normalize data to between 0 and 1
        train_features = train_features / 255

        # convert to pytorch tensors
        labels = torch.from_numpy(labels)
        train_features = torch.from_numpy(train_features)

        m = len(labels)
        print(m, ' training examples')

        # convert into dataloader
        trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_features, labels), batch_size=batch_size,
                                                  shuffle=True)
        # define neural network parameters
        # net = nn.Net()
        net = nn.ConNet()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
        optimizer.zero_grad(True)

        # train the neural network
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                re_inputs = inputs[0].view(1, 1, 28, 28)
                if isinstance(net, nn.ConNet):
                    for row in inputs[1:]:
                        re_inputs = torch.cat((re_inputs, row.view(1, 1, 28, 28)))
                    inputs = re_inputs
                else:
                    inputs = inputs.view(batch_size, 28*28)

                # .view(bs, 28 * 28) #inputs.view(bs, 1, 28, 28)
                pred = net(inputs)
                optimizer.zero_grad(True)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:  # print every 1000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 1000))
                    running_loss = 0.0

        print('Finished Training')

    else:
        net = torch.load(MODEL_TO_USE)
        net.eval()
        print('Model Loaded')
    torch.save(net, save_name)
    return net


def predict_all():
    training = pandas.read_csv('./data/test.csv')

    features = training.iloc[:, 0:].values
    test_features = np.array(features).reshape((len(features), 28, 28)).astype(np.single)
    test_features = torch.from_numpy(test_features)

    print(len(test_features))
    predicts = [[]]
    for i in range(len(test_features)):
        pred = predict(test_features[i])
        predicts.append([pred])
    predicts = pandas.DataFrame(predicts)
    predicts.to_csv('./predictions.csv',header=['Label'],float_format='%.0f')


if __name__ == "__main__":
    main()
