# ==============================================================================
# BÀI TẬP: MÔ HÌNH LENET-5 PHÂN LOẠI FASHION MNIST (PYTORCH - 50 EPOCHS)
# - LOAD TOÀN BỘ DỮ LIỆU LÊN GPU VRAM TRƯỚC KHI TRAIN (~4-5 GIÂY / 50 EPOCHS)
# - HIỂN THỊ CHI TIẾT LOSS VÀ ACCURACY CỦA CẢ TRAIN VÀ VALIDATION
# - TỰ ĐỘNG LƯU MODEL TỐT NHẤT, ĐÁNH GIÁ TRÊN TẬP TEST VÀ VẼ BIỂU ĐỒ
# ==============================================================================

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. THIẾT LẬP THIẾT BỊ (GPU)
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Đang sử dụng thiết bị: {device}")
torch.manual_seed(42)

# ------------------------------------------------------------------------------
# 2. TẢI VÀ LOAD TOÀN BỘ DỮ LIỆU LÊN GPU VRAM
# ------------------------------------------------------------------------------
print("⏳ Đang tải Fashion MNIST và đưa toàn bộ vào VRAM GPU...")

raw_train = datasets.FashionMNIST(root='./data', train=True, download=True)
raw_test = datasets.FashionMNIST(root='./data', train=False, download=True)

# Chuẩn hóa về [-1, 1] và thêm chiều channel: (N, 28, 28) -> (N, 1, 28, 28)
X_all = ((raw_train.data.float() / 255.0 - 0.5) / 0.5).unsqueeze(1)
y_all = raw_train.targets

X_test = ((raw_test.data.float() / 255.0 - 0.5) / 0.5).unsqueeze(1)
y_test = raw_test.targets

# Chia 50,000 ảnh Train và 10,000 ảnh Validation
X_train = X_all[:50000]
y_train = y_all[:50000]
X_val = X_all[50000:]
y_val = y_all[50000:]

# 🚀 NẠP HẾT TOÀN BỘ LÊN BỘ NHỚ VRAM CỦA GPU
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

print(f"✅ Đã nạp thành công lên GPU!")
print(f" • Kích thước Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ------------------------------------------------------------------------------
# 3. ĐỊNH NGHĨA KIẾN TRÚC MẠNG TÍCH CHẬP LENET-5
# ------------------------------------------------------------------------------
class LeNet5(nn.Module):
    """
    Kiến trúc mạng LeNet-5 kinh điển (Yann LeCun):
    - C1: Lớp Tích chập (Conv2d 1 -> 6 filters, 5x5) + ReLU
    - S2: Lớp Gộp cực đại (MaxPool2d 2x2, stride 2) -> (6, 14, 14)
    - C3: Lớp Tích chập (Conv2d 6 -> 16 filters, 5x5) + ReLU -> (16, 10, 10)
    - S4: Lớp Gộp cực đại (MaxPool2d 2x2, stride 2) -> (16, 5, 5)
    - C5: Fully Connected (16*5*5 = 400 -> 120) + Dropout
    - F6: Fully Connected (120 -> 84) + Dropout
    - Output: Fully Connected (84 -> 10 lớp phân loại)
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Khối C1 & S2
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2) # Giữ kích thước 28x28
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        
        # Khối C3 & S4
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # 14x14 -> 10x10
        self.bn2 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2) # 10x10 -> 5x5
        
        # Tầng Phân loại (Fully Connected)
        self.flatten = nn.Flatten()
        
        # C5
        self.c5 = nn.Linear(16 * 5 * 5, 120)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        
        # F6
        self.f6 = nn.Linear(120, 84)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)
        
        # Output
        self.out = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.s2(self.relu1(self.bn1(self.c1(x))))
        x = self.s4(self.relu3(self.bn2(self.c3(x))))
        x = self.flatten(x)
        x = self.dropout1(self.relu5(self.c5(x)))
        x = self.dropout2(self.relu6(self.f6(x)))
        x = self.out(x)
        return x

model = LeNet5().to(device)
print(model)

# ------------------------------------------------------------------------------
# 4. HÀM MẤT MÁT, OPTIMIZER VÀ SCHEDULER
# ------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ------------------------------------------------------------------------------
# 5. TIẾN HÀNH HUẤN LUYỆN 50 EPOCHS TRỰC TIẾP TRÊN GPU
# ------------------------------------------------------------------------------
EPOCHS = 50
BATCH_SIZE = 256
CHECKPOINT_PATH = "best_lenet5_model.pth"

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

best_val_acc = 0.0
num_train = X_train.size(0)

print(f"\n🚀 Bắt đầu huấn luyện LeNet-5 ({EPOCHS} epochs) trên GPU:\n")
total_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()

    # --- TRAIN ---
    model.train()
    running_train_loss, correct_train = 0.0, 0
    permutation = torch.randperm(num_train, device=device)

    for i in range(0, num_train, BATCH_SIZE):
        batch_indices = permutation[i : i + BATCH_SIZE]
        images = X_train[batch_indices]
        labels = y_train[batch_indices]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        correct_train += (preds == labels).sum().item()

    train_loss = running_train_loss / num_train
    train_acc = correct_train / num_train

    # --- VALIDATION (Tính toán song song toàn bộ tập Val) ---
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_preds = val_outputs.argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean().item()

    # Lưu lịch sử
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Cập nhật Scheduler
    scheduler.step(val_acc)
    epoch_duration = time.time() - epoch_start_time

    # --- IN GIÁ TRỊ TỪNG EPOCH ---
    print(f"Epoch [{epoch:02d}/{EPOCHS}] ({epoch_duration:.2f}s) | "
          f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%", end="")

    # Lưu lại model có Val Acc tốt nhất
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f" -> 💾 [Lưu Best LeNet-5: {val_acc*100:.2f}%]")
    else:
        print("")

total_time = time.time() - total_start_time

# ------------------------------------------------------------------------------
# 6. NẠP LẠI MODEL TỐT NHẤT VÀ ĐÁNH GIÁ TRÊN TẬP TEST
# ------------------------------------------------------------------------------
print(f"\n>>> Đang nạp lại mô hình LeNet-5 tốt nhất từ '{CHECKPOINT_PATH}'...")
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test).item()
    test_preds = test_outputs.argmax(dim=1)
    test_acc = (test_preds == y_test).float().mean().item()

print("\n" + "="*70)
print("🏆 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LENET-5:")
print("="*70)
print(f" • ⏱️ Tổng thời gian chạy 50 Epochs       : {total_time:.2f} giây")
print(f" • File model tốt nhất đã lưu            : '{CHECKPOINT_PATH}'")
print(f" • Độ chính xác lớn nhất trên tập Val   : {best_val_acc * 100:.2f}%")
print(f" • 🎯 ĐỘ CHÍNH XÁC TRÊN TẬP TEST        : {test_acc * 100:.2f}%")
print(f" • Mất mát (Loss) trên tập Test         : {test_loss:.4f}")
print("="*70 + "\n")

# ------------------------------------------------------------------------------
# 7. VẼ BIỂU ĐỒ LOSS VÀ ACCURACY (TRAIN vs VALIDATION)
# ------------------------------------------------------------------------------
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(14, 5))

# Đồ thị Độ chính xác
plt.subplot(1, 2, 1)
plt.plot(epochs_range, [acc*100 for acc in history['train_acc']], label='Train Accuracy', color='blue', linewidth=2)
plt.plot(epochs_range, [acc*100 for acc in history['val_acc']], label='Val Accuracy', color='green', linewidth=2)
plt.title('LeNet-5: Độ chính xác (Accuracy %) qua 50 Epochs', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Độ chính xác (%)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Đồ thị Hàm mất mát
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='green', linewidth=2)
plt.title('LeNet-5: Hàm mất mát (Loss) qua 50 Epochs', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()