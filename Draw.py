import matplotlib.pyplot as plt
from pandas import DataFrame
def Draw_loss(DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(DataFrame['epoch'], DataFrame['Loss'], label='MM-DBR')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('Value')
    plt.savefig('./img/loss.png')
    plt.show()
    return

def Draw_acc(DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(DataFrame['epoch'], DataFrame['Acc'], label='MM-DBR')
    plt.title('Acc')
    plt.xlabel('epoch')
    plt.ylabel('Value')
    plt.savefig('./img/acc.png')
    plt.show()
    return