import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models
import os
import glob
import random
import numpy as np
import pandas as pd
from datetime import datetime


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ResNet18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18_Manual(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet18_Manual, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TensorChunkDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.long)


def load_all_chunks(folder_path, prefix):
    search_pattern = os.path.join(folder_path, f"{prefix}*.pt")
    file_list = sorted(glob.glob(search_pattern))
    if not file_list: raise FileNotFoundError(f"File not found: {search_pattern}")
    return ConcatDataset([TensorChunkDataset(f) for f in file_list])



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return total_loss / total, 100 * correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return total_loss / total, 100 * correct / total



def run_single_experiment(config, train_loader, val_loader, device):
    print(f"\n{'=' * 60}")
    print(f"Start: {config['name']}")
    print(f"{'=' * 60}")

    # 初始化模型
    if config['model_type'] == 'SimpleCNN':
        model = SimpleCNN(num_classes=2).to(device)
    elif config['model_type'] == 'ResNet18_Manual':
        model = ResNet18_Manual(BasicBlock, layers=[2, 2, 2, 2], num_classes=2).to(device)
    else:
        raise ValueError("Unknown model type")

    # 初始化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_val_acc = 0.0
    start_time = datetime.now()

    # 训练循环
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(f"[{config['name']}] Epoch {epoch + 1:02d} -> Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # 记录结果
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    result = {
        "Experiment Name": config['name'],
        "Model Type": config['model_type'],
        "Learning Rate": config['lr'],
        "Epochs": config['epochs'],
        "Best Val Acc (%)": round(best_val_acc, 2),
        "Training Time (s)": round(duration, 1)
    }

    print(f"\nExperiment completed: {config['name']} | Best val accuracy: {best_val_acc:.2f}%")
    return result


if __name__ == "__main__":
    data_path = r"D:\Desktop\EE6483 Project\datasets\processed"
    batch_size = 32

    print("Loading data...")
    train_dataset = load_all_chunks(data_path, "train_part")
    val_dataset = load_all_chunks(data_path, "val_part")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Successfully loaded: Train={len(train_dataset)}, Val={len(val_dataset)}")

    experiment_configs = [
        # Simple CNN
        {
            "name": "Exp1_SimpleCNN_LR0.001",
            "model_type": "SimpleCNN",
            "lr": 0.001,
            "epochs": 15
        },

        # ResNet18
        {
            "name": "Exp2_ResNet18_Manual_LR0.001",
            "model_type": "ResNet18_Manual",
            "lr": 0.001,
            "epochs": 15
        },

    ]

    # 运行
    all_results = []
    for config in experiment_configs:
        try:
            result = run_single_experiment(config, train_loader, val_loader, device)
            all_results.append(result)
        except Exception as e:
            print(f"Experiment {config['name']} error: {e}")

    # 保存最终结果
    if all_results:
        df_results = pd.DataFrame(all_results)
        output_csv = "experiment_results.csv"
        df_results.to_csv(output_csv, index=False)

        print("\n" + "=" * 60)
        print("All completed:")
        print("=" * 60)
        print(df_results)
        print(f"\nResult: {os.path.abspath(output_csv)}")