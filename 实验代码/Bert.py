import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW,get_scheduler ,BertForSequenceClassification
from transformers import AutoTokenizer
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
import jieba


# 初始化日志
logging.basicConfig(filename='Bert_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().split('\t', 1)
            data.append((text, int(label)))  # 将标签转换为整数类型
    return data

# 使用jieba分词
def jieba_cut(text):
    return ' '.join(jieba.cut(text))

train_data = load_data('./data/train.txt')
dev_data = load_data('./data/dev.txt')
test_data = load_data('./data/test.txt')

# 分词并转换为DataFrame，方便查看和操作
train_data = [(jieba_cut(text), label) for text, label in train_data]
dev_data = [(jieba_cut(text), label) for text, label in dev_data]
test_data = [(jieba_cut(text), label) for text, label in test_data]

train_df = pd.DataFrame(train_data, columns=['text', 'label'])
dev_df = pd.DataFrame(dev_data, columns=['text', 'label'])
test_df = pd.DataFrame(test_data, columns=['text', 'label'])

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

logger.info(f"\nTraining set size: {len(train_data)}")
logger.info(f"Validation set size: {len(dev_data)}")
logger.info(f"Test set size: {len(test_data)}")

# 使用Bert - base - chinese的Tokenizer进行分词
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
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
            return_tensors='pt'  # Return pytorch tensors
        )
        # 打印输入数据的维度
        # labels shape: ()
        # Input IDs shape: torch.Size([1, 128])
        # Attention Mask shape: torch.Size([1, 128])
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 定义最大文本长度
max_len = 128
batch_size = 32

train_dataset = NewsDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataset = NewsDataset(dev_df['text'].tolist(), dev_df['label'].tolist(), tokenizer, max_len)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_dataset = NewsDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, max_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 加载BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 如果有多个GPU，使用DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 输出模型结构
logger.info("Model:\n{}".format(model))


# 定义优化器、学习率调度器、和损失函数
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=10*len(train_loader),
)

# 用于记录训练和验证过程中的准确率和损失值
# 画loss曲线
train_losses = []
train_accuracies = []
dev_losses = []
dev_accuracies = []
def train_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

        # 计算准确率
        _, predictions = torch.max(logits, dim=1)
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    return avg_loss, accuracy


def evaluate(model, data_loader, device, save_conf_matrix=False, save_dir='results-bert'):
    model.eval()
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            # 计算准确率
            _, predictions = torch.max(logits, dim=1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(data_loader)
    report = classification_report(all_labels, all_preds, digits=4)
    
    # if save_conf_matrix:
    #     cm = confusion_matrix(all_labels, all_preds)
    #     plt.figure(figsize=(10,7))
    #     sns.heatmap(cm, annot=True, fmt='d')
    #     plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    
    # return avg_loss, accuracy, report

    # 计算精确率
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
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
    train_avg_loss, train_accuracy = train_epoch(model, train_loader, optimizer,  device, epoch)
    logger.info(f"第 {epoch + 1} 轮训练平均损失: {train_avg_loss:.4f}, 训练准确率: {train_accuracy:.4f}")
    
    # 每隔2个epoch进行验证
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
if not os.path.exists('results—bert'):
    os.makedirs('results—bert')
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
plt.savefig('results—bert/train_val_curves.png')
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