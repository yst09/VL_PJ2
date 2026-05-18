import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import logging
from datetime import datetime

def setup_logger(log_file_path):
    """配置日志，使其同时输出到控制台和文件"""
    # 确保之前没有残留的 handler
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='resnet18, resnet34, vit_tiny_patch16_224')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrain')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for head')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs (with early stopping)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--drop_rate', type=float, default=0.0, help='Dropout rate for the model')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()

    # 1. 设置实验名称与日志文件
    # 使用时间戳确保日志不会因为重复运行被完全覆盖（可选）
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_pre_{args.pretrained}_lr_{args.lr}_drop_{args.drop_rate}_wd_{args.weight_decay}"
    
    # 确保有一个 logs 文件夹存放日志
    os.makedirs("logs", exist_ok=True)
    log_file_path = os.path.join("logs", f"{run_name}.log")
    
    logger = setup_logger(log_file_path)
    
    # 将实验配置记录到日志文件开头
    logger.info("="*50)
    logger.info(f"Starting new experiment: {run_name}")
    logger.info("Experiment Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 2. 数据增强与加载
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.Flowers102(root='../data', split='train', download=True, transform=train_transform)
    val_dataset = datasets.Flowers102(root='../data', split='val', download=True, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. 模型构建
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=102, drop_rate=args.drop_rate)
    model = model.to(device)

    # 4. 优化器设置
    if args.pretrained:
        head_params = model.get_classifier().parameters()
        head_params_ids = list(map(id, head_params))
        base_params = filter(lambda p: id(p) not in head_params_ids, model.parameters())
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': args.lr * 0.1},
            {'params': head_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
        logger.info("Configured AdamW optimizer for Fine-tuning (head gets full LR, base gets 0.1x LR)")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        logger.info("Configured AdamW optimizer for training from scratch")

    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环与早停机制
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # 确保有一个 weights 文件夹存放模型
    os.makedirs("weights", exist_ok=True)
    best_model_path = os.path.join("weights", f"best_{run_name}.pth")

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_acc = correct / len(val_dataset)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 将 Loss 和 Acc 写入日志
        logger.info(f"Epoch [{epoch+1:03d}/{args.epochs:03d}] | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}")

        # --- 早停与最佳模型判断逻辑 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"*** New best model saved! Acc: {best_val_acc:.4f} ***")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement. Patience: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            logger.warning(f"Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}")
            break

    logger.info(f"Experiment finished. Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info("="*50 + "\n")

if __name__ == "__main__":
    main()