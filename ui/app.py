import streamlit as st
from services.storage import store_documents
from services.rag import rag_chain

def main():
    st.set_page_config(page_title="RAG System", layout="wide")
    st.title("RAG (ES+FAISS+Deepseek-R1:14b)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("文件上传")
        file_path = st.text_input("输入文件夹路径", value="/teamspace/studios/this_studio/elasticsearch-Faiss-Deepseek-R1-14B-RAG/AAA-database")
        if st.button("上传"):
            store_documents(file_path)
            st.success("文件上传成功！")
        
        st.subheader("提问")
        question = st.text_area("输入问题", height=150)
        if st.button("提交"):
            if question:
                with st.spinner("生成回答中..."):
                    answer_container = col2.empty()
                    docs_container = col2.empty()
                    for answer, docs in rag_chain(question):
                        answer_container.text_area("回答", value=answer, height=200)
                        docs_container.write(format_documents(docs))

    with col2:
        st.subheader("结果展示")
        # 这里可以添加文件列表展示逻辑

if __name__ == "__main__":
    main()