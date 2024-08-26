import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.tensorboard import SummaryWriter
import torchviz
from dataset import MMDBR_Dataset
from Network import MMDBR_LeNet
from torch.utils.data import Dataset, DataLoader
from Draw import *
import sys
import signal
def train(model, tensor_loader, num_epochs, learning_rate, criterion, device, writer, test_loader):
    df = DataFrame(columns=['Acc','Loss',"epoch"])
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    signal.signal(signal.SIGINT, create_signal_handler(df, model, test_loader, criterion, device, writer, epoch=num_epochs))
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        best_accuracy = 0
        for data in tensor_loader:
            # print(data)
            inputs, labels = data
            # print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)

            correct_predictions = (predict_y == labels).sum().item()
            epoch_accuracy += correct_predictions / inputs.size(0)
            # epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(model.state_dict(), './model_pth/model_best.pth')
        writer.add_scalar('Loss/train', epoch_loss / len(tensor_loader.dataset), epoch)
        writer.add_scalar('Accuracy/train', epoch_accuracy / len(tensor_loader), epoch)
        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        df.loc[len(df)] = [epoch_accuracy, epoch_loss, epoch]
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))
        if epoch % 25 == 0:
            test(model, test_loader, criterion, device, writer, epoch=epoch)
    test(model, test_loader, criterion, device, writer, epoch=num_epochs)
    Draw_loss(df)
    Draw_acc(df)
    return


def test(model, tensor_loader, criterion, device, writer, epoch):
    model.eval()
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)

        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)

        loss = criterion(outputs, labels).to(device)
        predict_y = torch.argmax(outputs, dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)

    writer.add_scalar('Accuracy/test', test_acc, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
    return

def create_signal_handler(df, model, test_loader, criterion, device, writer, epoch):
    def signal_handler(sig, frame):
        print('Model printout!')
        interrupt_handler(df, model, test_loader, criterion, device, writer, epoch)
    return signal_handler

def interrupt_handler(df, model, test_loader, criterion, device, writer, epoch):
    test(model, test_loader, criterion, device, writer, epoch)
    Draw_loss(df)
    Draw_acc(df)
    sys.exit(0)
    return
def main():

    root = './Data/'
    train_loader = torch.utils.data.DataLoader(dataset=MMDBR_Dataset(root + 'MM-DBR/train_amp/'), batch_size=64,
                                               shuffle=True,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=MMDBR_Dataset(root + 'MM-DBR/test_amp/'), batch_size=64,
                                              shuffle=False,drop_last=True)
    model = MMDBR_LeNet(num_classes=6)
    train_epoch = 400
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0")
    model.to(device)

    writer = SummaryWriter()
    print('Start training...')
    train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=train_epoch,
        learning_rate=1e-4,
        criterion=criterion,
        device=device,
        writer=writer,
        test_loader=test_loader
    )
    # test(
    #     model=model,
    #     tensor_loader=test_loader,
    #     criterion=criterion,
    #     device=device,
    #     writer=writer
    # )
    writer.close()
    return


if __name__ == "__main__":
    main()
