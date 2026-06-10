import streamlit as st

from src.predict import predict_text


st.set_page_config(page_title="虚假评论/水军识别", layout="centered")

st.title("虚假评论/水军识别演示")

text = st.text_area(
    "输入评论文本",
    value="这家店太好了，姐妹们冲，五星好评返现，绝对不亏",
    height=140,
)

if st.button("开始识别", type="primary"):
    try:
        prediction = predict_text(text)
        st.metric("识别结果", prediction["result"])
        if prediction["score"] is not None:
            st.write(f"模型分数：{prediction['score']:.4f}")
        st.caption("label=1 表示疑似虚假评论/水军评论，label=0 表示正常评论。")
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("请先运行：python -m src.train --data data/sample_reviews.csv")

st.divider()
st.subheader("适合视频展示的演示流程")
st.write("1. 展示数据样例与标签含义。")
st.write("2. 运行训练脚本，展示评估报告和混淆矩阵。")
st.write("3. 打开本页面，输入正常评论、水军评论、边界案例各 1 条。")
st.write("4. 解释模型为什么可能判断正确或错误。")
