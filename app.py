import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

# ১. পেজ কনফিগারেশন ও প্রিমিয়াম ডিজাইন (CSS)
st.set_page_config(page_title="Class X History Smart AI Hub", page_icon="📚", layout="wide")

st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%); }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white; border-radius: 12px; font-weight: bold; border: none; padding: 10px 24px;
    }
    .stTextInput>div>div>input { border-radius: 12px; }
    .badge { background-color: #e0e7ff; color: #4338ca; padding: 4px 12px; border-radius: 8px; font-weight: bold; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Class X Social Studies (History) Smart AI Hub")
st.caption("🤖 Powered by Free Hugging Face AI Model & Streamlit Cloud — Commercial Premium Edition")

# ২. Hugging Face ফ্রি টোকেন সেটআপ
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Secrets থেকে PDF এর নাম নেওয়া (Advanced Settings এ যা সেট করেছেন)
PDF_FILE_NAME = os.getenv("PDF_FILE_NAME", "SocialScience English History Part-I Class X.pdf")

# ৩. পিডিএফ প্রসেসিং এবং ডাটাবেস তৈরি
@st.cache_resource
def initialize_knowledge_base(pdf_path):
    if not os.path.exists(pdf_path):
        return None
    
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

vector_db = initialize_knowledge_base(PDF_FILE_NAME)

# ৪. ইন্টারফেস ডিজাইন (বাম পাশের ফিল্টার প্যানেল)
with st.sidebar:
    st.header("⚙️ স্মার্ট কন্ট্রোল প্যানেল")
    st.write("আপনার পিডিএফ থেকে নির্দিষ্ট নম্বরের নির্ভুল উত্তর তৈরির গাইড।")
    
    selected_mark = st.radio(
        "%s" % "প্রশ্নের নম্বর (Marks) সিলেক্ট করুন:",
        ["1 Mark (১-২ লাইন)", "2 Marks (২-৩ লাইন)", "3 Marks (৩-৫ লাইন)", "5 Marks (১০০-২০০ শব্দ)"]
    )
    
    st.info("💡 সবকটি উত্তর বোর্ড এক্সামের খাতা দেখার নিয়ম মেনে পয়েন্ট আকারে এবং বাংলা অনুবাদসহ তৈরি হবে।")

# ডানপাশের মেইন স্ক্রিন
if vector_db is None:
    st.error(f"❌ ব্যাকএন্ডে '{PDF_FILE_NAME}' ফাইলটি পাওয়া যায়নি! দয়া করে আপনার ফাইলের নাম চেক করুন।")
else:
    query = st.text_input("🔍 প্রশ্ন বা কি-ওয়ার্ড টাইপ করুন:", placeholder="যেমন: Rowlatt Act, Jallianwala Bagh, Partition...")

    if st.button("🚀 সঠিক উত্তর তৈরি করুন"):
        if query:
            with st.spinner("সিলেবাসের নিয়মাবলী যাচাই এবং অনুবাদ প্রসেস করা হচ্ছে..."):
                try:
                    related_docs = vector_db.similarity_search(query, k=3)
                    context = "\n".join([doc.page_content for doc in related_docs])
                    
                    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
                    llm = HuggingFaceEndpoint(
                        repo_id=repo_id,
                        huggingfacehub_api_token=hf_token,
                        temperature=0.3,
                        max_new_tokens=1024
                    )
                    
                    prompt = f"""
                    You are an expert Class X History Teacher. Base your answer strictly on the context provided below.
                    Context: {context}
                    
                    Question: {query}
                    Target Marks Requirement: {selected_mark}
                    
                    Strict Formatting Rules:
                    1. If the requirement is '1 Mark', provide exactly 1 to 2 lines of clear answer.
                    2. If '2 Marks', provide 2 to 3 lines or 2 short points.
                    3. If '3 Marks', provide 3 to 5 lines or 3 points.
                    4. If '5 Marks', provide a detailed structured essay answer of 100 to 200 words with clear sub-headings/points.
                    5. First write the answer in high-quality 'Standard English Answer'.
                    6. Right below the English answer, write 'বাংলা অনুবাদ (Bengali Version)' containing an accurate, easy-to-understand Bengali translation for students.
                    
                    Output Layout:
                    [English Answer here]
                    
                    ---
                    ### 🇮🇳 বাংলা অনুবাদ (Bengali Version)
                    [Bengali Translation here]
                    """
                    
                    ai_response = llm.invoke(prompt)
                    
                    st.markdown(f"<span class='badge'>{selected_mark} এর উত্তর কাঠামো</span>", unsafe_allow_html=True)
                    st.subheader(f"❓ প্রশ্ন: {query}")
                    st.write(ai_response)
                    
                except Exception as e:
                    st.error("এআই মডেল রেসপন্স করতে পারছে না। আপনার Secrets এ টোকেনটি ঠিকমতো দেওয়া আছে কি না চেক করুন।")
        else:
            st.warning("দয়া করে প্রথমে একটি প্রশ্ন টাইপ করুন!")
