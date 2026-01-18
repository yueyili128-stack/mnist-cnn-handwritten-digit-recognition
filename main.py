import torch
from models.cnn import CNN
from utils.dataset import get_dataloaders
from scripts.train import train
from scripts.evaluate import evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    train_loader, test_loader = get_dataloaders()

    train(model, train_loader, device)
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
