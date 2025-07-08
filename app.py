import os
import torch
from torch.utils.data import DataLoader
from model import EffB0
from torchvision import transforms
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置页面配置
st.set_page_config(
    page_title="狗品种分类",
    page_icon="🐶"
)
# import os
# import torch
# from torch.utils.data import DataLoader
# from model import EffB0  # 确保model.py文件在同一目录下
# from torchvision import transforms
# import streamlit as st
# import pandas as pd
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei']  # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置页面配置
# st.set_page_config(
#     page_title="food-101分类",
#     page_icon="📷"
# )

# 页面美化：顶部横幅
st.markdown("""
    <div style='background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); padding: 24px 0 12px 0; border-radius: 8px; margin-bottom: 18px;'>
        <h1 style='color: white; text-align: center; margin-bottom: 0;'>你的狗狗我知道</h1>
        <p style='color: #e0e0e0; text-align: center; margin-top: 8px;'>上传一张图片，了解你的狗狗品种</p>
    </div>
    <hr style='margin-bottom: 24px;'/>
""", unsafe_allow_html=True)

# # 1. 自动获取food-101类别顺序
# FOOD101_ROOT = "E:/food-101/food-101"  # 这里改成你的food-101根目录
# CLASSES_TXT = os.path.join(FOOD101_ROOT, 'meta', 'classes.txt')
# with open(CLASSES_TXT, 'r', encoding='utf-8') as f:
#     classes = [line.strip() for line in f.readlines()]

# 1. 读取训练时保存的类别顺序
with open('classes.txt', 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

# 2. 英文到中文映射（可补充更多狗品种）
en2zh = {
    "Shiba_Dog": "柴犬",
    "French_bulldog": "法国斗牛犬",
    "Siberian_husky": "西伯利亚哈士奇",
    "malamute": "阿拉斯加雪橇犬",
    "Pomeranian": "博美犬",
    "Airedale": "艾尔戴尔梗",
    "miniature_poodle": "迷你贵宾犬",
    "affenpinscher": "阿芬宾犬",
    "schipperke": "斯奇派克犬",
    "Australian_terrier": "澳大利亚梗",
    "Welsh_springer_spaniel": "威尔士激飞猎犬",
    "curly_coated_retriever": "卷毛寻回犬",
    "Staffordshire_bullterrier": "斯塔福郡斗牛梗",
    "Norwich_terrier": "诺里奇梗",
    "Tibetan_terrier": "西藏梗",
    "English_setter": "英国雪达犬",
    "Norfolk_terrier": "诺福克梗",
    "Pembroke": "彭布罗克柯基",
    "Tibetan_mastiff": "藏獒",
    "Border_terrier": "边境梗",
    "Great_Dane": "大丹犬",
    "Scotch_terrier": "苏格兰梗",
    "flat_coated_retriever": "平毛寻回犬",
    "Saluki": "萨路基猎犬",
    "Irish_setter": "爱尔兰雪达犬",
    "Blenheim_spaniel": "布伦海姆猎犬",
    "Irish_terrier": "爱尔兰梗",
    "bloodhound": "寻血猎犬",
    "redbone": "红骨猎犬",
    "West_Highland_white_terrier": "西高地白梗",
    "Brabancon_griffo": "布拉班松格里芬犬",
    "dhole": "豺狗",
    "kelpie": "凯尔皮犬",
    "Doberman": "杜宾犬",
    "Ibizan_hound": "伊比赞猎犬",
    "vizsla": "维兹拉犬",
    "cairn": "凯恩梗",
    "German_shepherd": "德国牧羊犬",
    "African_hunting_dog": "非洲猎犬",
    "Dandie_Dinmont": "丹迪丁蒙特梗",
    "Sealyham_terrier": "西利汉梗",
    "German_short_haired_pointer": "德国短毛指示犬",
    "Bernese_mountain_dog": "伯恩山犬",
    "Saint_Bernard": "圣伯纳犬",
    "Leonberg": "莱昂贝格犬",
    "Bedlington_terrier": "贝德灵顿梗",
    "Newfoundland": "纽芬兰犬",
    "Lhasa": "拉萨犬",
    "Chesapeake_Bay_retriever": "切萨皮克湾寻回犬",
    "Lakeland_terrier": "莱克兰梗",
    "Walker_hound": "沃克猎犬",
    "American_Staffordshire_terrier": "美国斯塔福郡梗",
    "otterhound": "水獭猎犬",
    "Sussex_spaniel": "萨塞克斯猎犬",
    "Norwegian_elkhound": "挪威猎麋犬",
    "bluetick": "蓝色滴答猎犬",
    "dingo": "澳洲野犬",
    "Irish_water_spaniel": "爱尔兰水猎犬",
    "Samoyed": "萨摩耶",
    "Fila Braziliero": "巴西菲拉犬",
    "standard_schnauzer": "标准雪纳瑞",
    "Mexican_hairless": "墨西哥无毛犬",
    "EntleBucher": "恩特勒布赫山犬",
    "Afghan_hound": "阿富汗猎犬",
    "kuvasz": "库瓦兹犬",
    "English_foxhound": "英国猎狐犬",
    "keeshond": "荷兰毛狮犬",
    "Irish_wolfhound": "爱尔兰猎狼犬",
    "Scottish_deerhound": "苏格兰猎鹿犬",
    "Rottweiler": "罗威纳犬",
    "black_and_tan_coonhound": "黑褐浣熊猎犬",
    "Great_Pyrenees": "大比利牛斯山犬",
    "boxer": "拳师犬",
    "wire_haired_fox_terrier": "刚毛猎狐梗",
    "borzoi": "俄罗斯猎狼犬",
    "groenendael": "格罗安达尔犬",
    "collie": "柯利牧羊犬",
    "Gordon_setter": "戈登雪达犬",
    "Kerry_blue_terrier": "凯利蓝梗",
    "briard": "布里亚德牧羊犬",
    "Rhodesian_ridgeback": "罗得西亚脊背犬",
    "Boston_bull": "波士顿斗牛犬",
    "bull_mastiff": "斗牛獒犬",
    "silky_terrier": "丝毛梗",
    "Brittany_spaniel": "布列塔尼猎犬",
    "Eskimo_dog": "爱斯基摩犬",
    "giant_schnauzer": "巨型雪纳瑞",
    "malinois": "马里努阿犬",
    "Bouvier_des_Flandres": "弗兰德斯牧牛犬",
    "whippet": "惠比特犬",
    "Appenzeller": "阿彭策尔山犬",
    "Chinese_Crested_Dog": "中国冠毛犬",
    "miniature_schnauzer": "迷你雪纳瑞",
    "soft_coated_wheaten_terrier": "软毛麦色梗",
    "Weimaraner": "威玛猎犬",
    "clumber": "克伦伯犬",
    "Greater_Swiss_Mountain_dog": "大瑞士山地犬",
    "toy_terrier": "玩具梗",
    "Italian_greyhound": "意大利灵缇",
    "basset": "巴吉度猎犬",
    "basenji": "巴仙吉犬",
    "Australian_Shepherd": "澳大利亚牧羊犬",
    "Maltese_dog": "马尔济斯犬",
    "Japanese_spaniel": "日本狆",
    "Cane_Carso": "卡内科尔索犬",
    "Japanese_Spitzes": "日本斯皮茨犬",
    "Old_English_sheepdog": "英国古代牧羊犬",
    "Black_sable": "黑貂犬",
    "Border_collie": "边境牧羊犬",
    "Shetland_sheepdog": "谢德兰牧羊犬",
    "English_springer": "英国激飞猎犬",
    "beagle": "比格犬",
    "cocker_spaniel": "可卡犬",
    "Cardigan": "卡迪根威尔士柯基",
    "toy_poodle": "玩具贵宾犬",
    "Bichon_Frise": "比熊犬",
    "standard_poodle": "标准贵宾犬",
    "komondor": "科蒙多犬",
    "chow": "松狮犬",
    "chinese_rural_dog": "中华田园犬",
    "Yorkshire_terrier": "约克夏梗",
    "Labrador_retriever": "拉布拉多寻回犬",
    "Shih_Tzu": "西施犬",
    "Chihuahua": "吉娃娃",
    "Pekinese": "北京犬",
    "golden_retriever": "金毛寻回犬",
    "miniature_pinscher": "迷你品犬",
    "teddy": "泰迪犬",
    "pug": "巴哥犬",
    "papillon": "蝴蝶犬"
}
# 3. 索引到中文类别（用完整文件夹名的最后一段查字典）
idx_to_class = {i: en2zh.get(cls.split('-')[-1], cls) for i, cls in enumerate(classes)}

# 图像预处理转换
transforms_ = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    FIXED_MODEL_PATH = "Dog_model_augment.pth"  # 你的模型文件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EffB0(num_classes=len(classes)).to(device)
    model.baseline_extractor.classifier = torch.nn.Linear(in_features=1280, out_features=len(classes)).to(device)
    model.load_state_dict(torch.load(FIXED_MODEL_PATH, map_location=device))
    st.success("模型加载成功！")
    return model, device

def predict(model, device, image):
    if model is None:
        return None, None, None
    image = image.convert('RGB')
    image_tensor = transforms_(image)
    image_tensor = image_tensor.reshape(1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2])
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        outputs, _ = model(image_tensor)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        _, predicted = torch.max(confidence, 0)
    predicted_class = int(predicted.item())
    confidence_value = float(confidence[predicted_class].item())
    return predicted_class, confidence_value, confidence.cpu().numpy()

def get_dog_wiki(dog_name, api_key):
    prompt = f"请用简明中文介绍一下{dog_name}这种狗的品种特征、性格、适合人群等。"
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def main():
    model, device = load_model()
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png", "webp"], help="支持jpg、jpeg、png、webp格式")
    if model is None or device is None:
        st.warning("模型加载出现问题，请检查模型路径和环境依赖")
        return
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown("<div style='background:#f8f9fa;padding:18px 18px 18px 18px;border-radius:8px;box-shadow:0 2px 8px #0001; margin-bottom: 18px;'>", unsafe_allow_html=True)
        st.subheader("上传的图片", divider="rainbow")
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='background:#e3f2fd;padding:18px 18px 18px 18px;border-radius:8px;box-shadow:0 2px 8px #0001; margin-bottom: 18px;'>", unsafe_allow_html=True)
        st.subheader("预测结果", divider="rainbow")
        # 预测按钮
        if st.button("开始预测", use_container_width=True):
            with st.spinner("正在分析图像..."):
                result = predict(model, device, image)
            if result is not None:
                predicted_class, confidence, all_confidences = result
                class_name = idx_to_class.get(predicted_class, f"未知类别 {predicted_class}")
                st.session_state["predicted_class"] = predicted_class
                st.session_state["confidence"] = confidence
                st.session_state["all_confidences"] = all_confidences
                st.session_state["class_name"] = class_name
                st.session_state["dog_info"] = None
                st.session_state["show_modal"] = False

        # 显示预测结果
        if "predicted_class" in st.session_state and st.session_state["predicted_class"] is not None:
            class_name = st.session_state["class_name"]
            confidence = st.session_state["confidence"]
            st.markdown(f"<h2 style='color:#1976d2;text-align:center;'>预测类别: <b>{class_name}</b></h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#333;text-align:center;'>置信度: <b>{confidence:.2f}%</b></h3>", unsafe_allow_html=True)
            if confidence is not None:
                st.progress(confidence / 100)
            # 狗狗百科按钮
            if st.button("狗狗百科", key="dog_wiki"):
                api_key = st.secrets["DEEPSEEK_API_KEY"]
                with st.spinner("正在召唤AI百科，请稍候..."):
                    try:
                        dog_info = get_dog_wiki(class_name, api_key)
                        st.session_state["dog_info"] = dog_info
                        st.session_state["show_modal"] = True
                    except Exception as e:
                        st.session_state["dog_info"] = f"百科查询失败: {e}"
                        st.session_state["show_modal"] = True

        st.markdown("</div>", unsafe_allow_html=True)

        # 兼容方案：直接在页面显示百科内容
        if st.session_state.get("show_modal", False):
            st.info(st.session_state.get("dog_info", "百科内容获取失败"))
            if st.button("关闭百科", key="close_modal"):
                st.session_state["show_modal"] = False

        # 概率分布（Top-N横向条形图）
        if "all_confidences" in st.session_state and st.session_state["all_confidences"] is not None and hasattr(st.session_state["all_confidences"], '__iter__'):
            st.divider()
            st.subheader("各类别概率分布（Top-10）", divider="rainbow")
            confidences = st.session_state["all_confidences"]
            N = 10  # Top-N
            top_indices = np.argsort(confidences)[-N:][::-1]
            top_labels = [idx_to_class[i] for i in top_indices]
            top_values = [confidences[i] for i in top_indices]

            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.barh(top_labels, top_values, color="#4b6cb7", alpha=0.85)
            ax.set_xlabel("置信度 (%)", fontsize=16)
            ax.set_title(f"Top-{N} 类别预测概率", fontsize=18, color="#4b6cb7")
            plt.yticks(fontsize=13)
            plt.xticks(fontsize=13)
            ax.invert_yaxis()  # 最高概率在最上面
            for bar in bars:
                width = bar.get_width()
                ax.annotate(f'{width:.1f}',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(5, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=10, color="#333")
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("请在左侧上传一张图片进行分类预测。", icon="📤")

if __name__ == "__main__":
    main()