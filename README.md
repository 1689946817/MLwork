东北大学机器学习课程作业



# 狗狗品种识别（Dog Breed Classification）

本项目基于清华大学 Tsinghua Dogs 数据集，利用深度学习方法实现对狗狗品种的自动识别。用户可通过 Web 页面上传狗狗图片，系统将自动识别其品种，并给出详细的概率分布，同时可以一键调用deepseek大模型来对狗狗的品种进行百科介绍。

---

## 项目亮点

- **数据集丰富**：采用 Tsinghua Dogs 数据集，涵盖130个常见犬种，图片数量超7万张，且包含头部和全身的标注框。
- **深度学习模型**：基于 EfficientNet-B0 架构，支持迁移学习和自定义训练。
- **Web 可视化**：使用 Streamlit 实现交互式网页，支持图片上传、预测结果展示、概率分布可视化及deepseek百科查询。
- **中文友好**：界面及输出均为中文，适合国内用户。

---

## 数据集介绍

- **名称**：Tsinghua Dogs Dataset  
- **来源**：[Tsinghua Dogs 官网](https://cg.cs.tsinghua.edu.cn/ThuDogs/)
- **类别数**：130
- **图片数**：70428
- **标注**：每张图片均有品种标签、狗头和全身的边界框
- **下载链接**：  
  - [低分辨率图片](https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/?dl=1)  
  - [低分辨率标注](https://cg.cs.tsinghua.edu.cn/ThuDogs/low-annotations.zip)  
  - [高分辨率图片及标注](https://cg.cs.tsinghua.edu.cn/ThuDogs/)  
- **数据集详细介绍**：[官方页面](https://cg.cs.tsinghua.edu.cn/ThuDogs/)

---

## 技术框架

- **深度学习框架**：PyTorch
- **模型结构**：EfficientNet-B0
- **前端展示**：Streamlit
- **可视化**：Matplotlib
- **辅助工具**：Pillow、NumPy、Requests

---

## 主要功能

1. **图片上传**：支持 JPG、JPEG、PNG、WEBP 格式。
2. **狗狗品种预测**：自动识别上传图片中的狗狗品种，并给出置信度。
3. **概率分布可视化**：以 Top-N 横向条形图展示预测概率分布。
4. **狗狗百科**：集成 DeepSeek 百科接口，自动生成品种介绍。
5. **中文界面**：所有输出均为中文，支持中文狗狗品种名。

---

## 目录结构

```
mlwork3/
├── app.py                # Streamlit 主程序
├── model.py              # EfficientNet-B0 模型定义
├── classes.txt           # 品种类别列表
├── Dog_model_augment.pth # 训练好的模型权重
└── ...                   # 其它辅助文件
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

请从[官网](https://cg.cs.tsinghua.edu.cn/ThuDogs/)下载低分辨率图片和标注，并解压到对应文件夹。

### 3. 训练模型

> 如使用已提供训练好的模型（Dog_model_augment.pth），可跳过本步。


```

### 4. 启动Web应用

```bash
streamlit run app.py
```

### 5. 使用方法

- 打开网页，上传一张狗狗图片，点击“开始预测”即可获得品种预测结果和概率分布。
- 可点击“狗狗百科”获取该品种的详细介绍。

---
### 6. 网页展示
![image](https://github.com/user-attachments/assets/255443e2-cfd2-4915-abb9-f93ad6092d30)
![image](https://github.com/user-attachments/assets/5653093a-61e6-4103-bd67-e09288d16970)
![image](https://github.com/user-attachments/assets/1ea0235a-8a4f-4be9-b685-a99a8b5e8555)
![image](https://github.com/user-attachments/assets/4179d3a2-8929-4e52-86d6-bf7445875b4b)
![image](https://github.com/user-attachments/assets/dcf5a851-f35f-4456-ad09-193291087a80)
![image](https://github.com/user-attachments/assets/8e6204fc-c353-4766-984d-b83aaada01aa)
![image](https://github.com/user-attachments/assets/b41cb406-3f20-4ea0-abff-09954e6f006d)

## 训练细节

- **模型结构**：EfficientNet-B0，最后全连接层根据类别数自适应调整。
- **损失函数**：交叉熵
- **优化器**：Adam
- **数据增强**：随机裁剪、翻转、归一化等
- **训练轮数**：建议50-200轮，根据实际情况调整
- **评测指标**：Top-1准确率、Top-5准确率

---

## 参考与致谢

- 数据集：[Tsinghua Dogs Dataset](https://cg.cs.tsinghua.edu.cn/ThuDogs/)
- 论文引用：
  ```
  @article{Zou2020ThuDogs,
    title={A new dataset of dog breed images and a benchmark for fine-grained classification},
    author={Zou, Ding-Nan and Zhang, Song-Hai and Mu, Tai-Jiang and Zhang, Min},
    journal={Computational Visual Media},
    year={2020},
    url={https://doi.org/10.1007/s41095-020-0184-6}
  }
  ```

---

## 常见问题

1. **如何更换模型结构？**  
   修改 `model.py`，替换为所需的网络结构，并调整 `app.py` 加载方式。

2. **如何添加新的狗狗品种？**  
   更新 `classes.txt`，并重新训练模型。

3. **如何自定义百科接口？**  
   修改 `get_dog_wiki` 函数，替换为你自己的API或本地百科。

---

## 联系方式

如有问题或建议，欢迎提 issue 或邮件联系作者。

---
