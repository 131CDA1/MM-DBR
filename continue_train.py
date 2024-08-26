from run import *

root = './Data/'
train_loader = torch.utils.data.DataLoader(dataset=MMDBR_Dataset(root + 'MM-DBR/train_amp/'), batch_size=64,
                                           shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=MMDBR_Dataset(root + 'MM-DBR/test_amp/'), batch_size=64,
                                          shuffle=False, drop_last=True)
model = MMDBR_LeNet(num_classes=6)
train_epoch = 300
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0")
model.to(device)

writer = SummaryWriter()
print('Start training...')

model_weights_path = './model_pth/model_best.pth'
model.load_state_dict(torch.load(model_weights_path))

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
writer.close()
