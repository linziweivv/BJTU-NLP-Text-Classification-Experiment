import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import csv
from transformers import GPT2Tokenizer, GPT2Config

# 初始化日志
logging.basicConfig(filename='Tranformer_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
    data['text'] = data['text'].str.strip()
    return data

train_data = load_data('./data/train.txt')
dev_data = load_data('./data/dev.txt')
test_data = load_data('./data/test.txt')

logger.info(f"\nTraining set size: {len(train_data)}")
logger.info(f"Validation set size: {len(dev_data)}")
logger.info(f"Test set size: {len(test_data)}")

# 使用GPT-2的Tokenizer进行分词
gpt2_config = GPT2Config.from_pretrained('./GPT2-124M')
gpt2_config.num_hidden_layers = 256
gpt2_config.output_attentions = True
gpt2_config.output_hidden_states = True
tokenizer = GPT2Tokenizer.from_pretrained('./GPT2-124M', trust_remote_code=True, local_files_only=True)

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 将pad_token设置为eos_token
logger.info("\nBERT tokenizer loaded\n")

# 定义Dataset类
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]


        # 使用BERT tokenizer编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # [CLS] and [SEP]
            max_length=self.max_len,
            padding='max_length',  # Padding to max_len
            truncation=True,  # Truncate if length exceeds max_len
            return_attention_mask=True,  # Attention mask
            return_tensors='pt',  # Return pytorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定义最大文本长度
MAX_LEN = 128
logger.info(f"\n最大文本长度设为: {MAX_LEN}")

# 创建训练集、验证集和测试集的DataLoader
train_dataset = NewsDataset(train_data['text'].values, train_data['label'].values, tokenizer, MAX_LEN)
dev_dataset = NewsDataset(dev_data['text'].values, dev_data['label'].values, tokenizer, MAX_LEN)
test_dataset = NewsDataset(test_data['text'].values, test_data['label'].values, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

import torch
import torch.nn as nn


# 定义仅基于Transformer的模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, n_heads, dropout):
        super(TransformerClassifier, self).__init__()
        # 嵌入层，将输入的token索引转换为向量表示
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, dropout=dropout),
            num_layers=n_layers
        )
        # 全连接层用于分类
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        # 获取嵌入后的向量表示，形状为(batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        # 调整维度顺序以匹配Transformer输入要求，变为(seq_len, batch_size, embedding_dim)
        embedded = embedded.permute(1, 0, 2)
        # 对嵌入向量添加位置编码
        seq_len = embedded.size(0)
        position_encoding = self.get_position_encoding(seq_len, embedded.size(2)).unsqueeze(1).to(embedded.device)
        embedded += position_encoding
        # 通过Transformer编码器
        transformer_output = self.transformer(embedded)
        # 取平均池化，形状变为(batch_size, hidden_dim)
        pooled_output = transformer_output.mean(dim=0)
        # 经过Dropout和全连接层得到最终输出
        output = self.fc(self.dropout(pooled_output))
        return output

    def get_position_encoding(self, seq_len, embedding_dim):
        """
        简单的位置编码函数参考Transformer原论文实现方式
        """
        position_encoding = torch.zeros(seq_len, embedding_dim)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        position_encoding[:, 0::2] = torch.sin(torch.arange(0, seq_len).unsqueeze(1).float() * div_term)
        position_encoding[:, 1::2] = torch.cos(torch.arange(0, seq_len).unsqueeze(1).float() * div_term)
        return position_encoding

# 定义模型参数
VOCAB_SIZE = tokenizer.vocab_size
HIDDEN_DIM = 768  
EMBEDDING_DIM = 768
OUTPUT_DIM = 10
N_LAYERS = 2
N_HEADS = 8
DROPOUT = 0.1

# 初始化模型
model = TransformerClassifier(VOCAB_SIZE,EMBEDDING_DIM ,HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, N_HEADS, DROPOUT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 如果有多个GPU，使用DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 输出模型结构
logger.info("Model:\n{}".format(model))

# 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# 用于记录训练和验证过程中的准确率和损失值
train_losses = []
train_accuracies = []
dev_losses = []
dev_accuracies = []

def train_epoch(model, data_loader, optimizer, criterion, device, epoch):
    model = model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # 使用 tqdm 显示进度条
    for i, batch in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False)):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        _, predictions = torch.max(outputs, dim=1)
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    return avg_loss, accuracy

def evaluate(model, data_loader, device, save_conf_matrix=False, save_dir='results'):
    model = model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask)

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 计算准确率
            _, predictions = torch.max(outputs, dim=1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(data_loader)
    report = classification_report(all_labels, all_preds, digits=4)

    # 计算精确率
    precision = precision_score(all_labels, all_preds, average='weighted')
    # 计算召回率
    recall = recall_score(all_labels, all_preds, average='weighted')
    # 计算F1 - score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 - score: {f1:.4f}")
    logger.info(f"Confusion Matrix: \n{conf_matrix}")

    if save_conf_matrix:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        img_file_name = os.path.join(save_dir, f'confusion_matrix_{data_loader.dataset.__class__.__name__}_{timestamp}.png')
        csv_file_name = os.path.join(save_dir, f'confusion_matrix_{data_loader.dataset.__class__.__name__}_{timestamp}.csv')

        plt.figure(figsize=(8, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(set(all_labels)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)

        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(img_file_name)
        plt.close()

        with open(csv_file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(conf_matrix)

    dev_losses.append(avg_loss)
    dev_accuracies.append(accuracy)
    return accuracy, report

# 训练和评估模型，添加日志记录和进度条
EPOCHS = 10
best_dev_accuracy = 0.0
patience = 3
no_improvement_count = 0
best_model_state_dict = None

for epoch in range(EPOCHS):
    logger.info(f"\n开始训练第 {epoch + 1} 轮：")

    # 训练阶段
    train_epoch(model, train_loader, optimizer, criterion, device, epoch)

    # 每隔 2 个 epoch 进行验证
    if (epoch + 1) % 2 == 0:
        logger.info(f"开始评估第 {epoch + 1} 轮：")
        dev_accuracy, dev_report = evaluate(model, dev_loader, device, save_conf_matrix=True)
        logger.info(f"第 {epoch + 1} 轮验证准确率: {dev_accuracy:.4f}")
        logger.info(f"验证集分类报告：\n{dev_report}")

        # 早停机制检查
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            no_improvement_count = 0
            # 保存当前最好模型的状态字典
            best_model_state_dict = model.state_dict()
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                logger.info(f"验证集指标连续 {patience} 次未提升，提前停止训练。")
                break

# 保存最好的模型
if best_model_state_dict is not None:
    model_path = "best_trained_model.pth"
    torch.save(best_model_state_dict, model_path)
    logger.info(f"最好的模型已保存至 {model_path}")

# 绘制并保存训练和验证的准确率、损失图表
if not os.path.exists('results'):
    os.makedirs('results')
plt.figure(figsize=(12, 4))

# 绘制训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(dev_losses, label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制训练准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(dev_accuracies, label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('results/train_val_curves.png')
plt.close()

# 加载最好的模型用于测试
if best_model_state_dict is not None:
    model.load_state_dict(best_model_state_dict)
    logger.info("加载最好的模型用于测试。")
else:
    logger.warning("没有找到最好的模型，使用训练后的最后一个模型进行测试。")

# 最终在测试集上进行评估
logger.info("\n开始在测试集上评估:")
test_accuracy, test_report = evaluate(model, test_loader, device, save_conf_matrix=True)
logger.info(f"测试集准确率: {test_accuracy:.4f}")
logger.info(f"测试集分类报告：\n{test_report}")