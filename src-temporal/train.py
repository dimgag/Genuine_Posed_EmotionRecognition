import torch 

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0
    train_acc_real_fake = 0
    train_acc_emotion = 0
    
    for batch in train_loader:
        videos, real_fake_labels, emotion_labels = batch
        videos = videos.to(device)
        real_fake_labels = real_fake_labels.to(device)
        emotion_labels = emotion_labels.to(device)
        
        optimizer.zero_grad()
        
        real_fake_preds, emotion_preds = model(videos)
        
        loss = criterion(real_fake_preds, real_fake_labels) + criterion(emotion_preds, emotion_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc_real_fake += accuracy(real_fake_preds, real_fake_labels)
        train_acc_emotion += accuracy(emotion_preds, emotion_labels)
    
    train_loss /= len(train_loader)
    train_acc_real_fake /= len(train_loader)
    train_acc_emotion /= len(train_loader)
    
    return train_loss, train_acc_real_fake, train_acc_emotion

def validate(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0
    test_acc_real_fake = 0
    test_acc_emotion = 0
    
    with torch.no_grad():
        for batch in test_loader:
            videos, real_fake_labels, emotion_labels = batch
            videos = videos.to(device)
            real_fake_labels = real_fake_labels.to(device)
            emotion_labels = emotion_labels.to(device)

            real_fake_preds, emotion_preds = model(videos)

            loss = criterion(real_fake_preds, real_fake_labels) + criterion(emotion_preds, emotion_labels)

            test_loss += loss.item()
            test_acc_real_fake += accuracy(real_fake_preds, real_fake_labels)
            test_acc_emotion += accuracy(emotion_preds, emotion_labels)

    test_loss /= len(test_loader)
    test_acc_real_fake /= len(test_loader)
    test_acc_emotion /= len(test_loader)
    
    return test_loss, test_acc_real_fake, test_acc_emotion

def accuracy(preds, labels):
    with torch.no_grad():
        _, preds = torch.max(preds, dim=1)
        correct = torch.sum(preds == labels)
        acc = correct.item() / len(labels)
        return acc
