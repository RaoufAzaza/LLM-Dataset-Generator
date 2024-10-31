import streamlit as st
import PyPDF2
import pandas as pd
import json
import re
from typing import List, Dict
import tempfile
from sklearn.model_selection import train_test_split
import requests

# Simplified model configs with only working models
MODEL_CONFIGS = {
    "Llama 3.2 (11B)": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "Mistral (7B)": "mistralai/Mistral-7B-Instruct-v0.2"
}

# Initialize session state
def init_session_state():
    if 'train_df' not in st.session_state:
        st.session_state.train_df = None
    if 'val_df' not in st.session_state:
        st.session_state.val_df = None
    if 'generated' not in st.session_state:
        st.session_state.generated = False
    if 'previous_upload_state' not in st.session_state:
        st.session_state.previous_upload_state = False

def reset_session_state():
    st.session_state.train_df = None
    st.session_state.val_df = None
    st.session_state.generated = False

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u00C0-\u017F.,?!]', '', text)
    text = re.sub(r'([.,?!])(\w)', r'\1 \2', text)
    return text.strip()

def parse_pdf(uploaded_file) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.seek(0)
            
            reader = PyPDF2.PdfReader(tmp_file.name)
            text_content = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                page_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', page_text)
                page_text = re.sub(r'(\w)-\n(\w)', r'\1\2', page_text)
                page_text = re.sub(r'\n+', ' ', page_text)
                text_content.append(clean_text(page_text))
            
            return ' '.join(text_content)
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return ""

def get_language_prompt(language: str, num_pairs: int, text: str) -> str:
    if language == "French":
        return f"""
        Générez {num_pairs} paires de questions-réponses à partir du texte suivant.

        Texte : {text}

        Consignes :
        1. Créez des questions variées
        2. Les réponses doivent être précises
        3. Évitez les questions simples ou répétitives
        4. Couvrez différents aspects du texte
        5. Utilisez un français correct

        Format : 
        Q: [Question]
        A: [Réponse]
        """
    else:
        return f"""
        Generate {num_pairs} question-answer pairs from the following text.

        Text: {text}

        Guidelines:
        1. Create varied questions
        2. Answers should be accurate
        3. Avoid simple or repetitive questions
        4. Cover different aspects of the text
        5. Use proper English

        Format:
        Q: [Question]
        A: [Answer]
        """

def parse_qa_response(response_text: str) -> List[Dict[str, str]]:
    qa_pairs = []
    pairs = re.split(r'\n\s*(?=Q:)', response_text)
    
    for pair in pairs:
        if 'Q:' in pair and 'A:' in pair:
            try:
                q_parts = pair.split('A:')
                question = q_parts[0].replace('Q:', '').strip()
                answer = q_parts[1].strip()
                
                if len(question) > 10 and len(answer) > 10:
                    qa_pairs.append({
                        'Question': clean_text(question),
                        'Answer': clean_text(answer)
                    })
            except Exception:
                continue
    
    return qa_pairs

def llm_api_call(messages, api_key: str, model_name: str):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.9,
        "stop": ["<|eot_id|>", "<|eom_id|>"],
        "messages": messages
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def generate_qa_pairs(text: str, api_key: str, model_name: str, num_pairs: int, language: str) -> pd.DataFrame:
    prompt = get_language_prompt(language, num_pairs, text)
    system_prompt = "You are an expert at generating high-quality question-answer pairs."

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = llm_api_call(messages, api_key, model_name)
        
        if not response or 'choices' not in response or not response['choices']:
            raise Exception("Invalid response")
        
        qa_text = response['choices'][0]['message']['content']
        qa_pairs = parse_qa_response(qa_text)
        
        if not qa_pairs:
            raise Exception("No valid QA pairs generated")
        
        df = pd.DataFrame(qa_pairs)
        df['Context'] = text
        return df
    
    except Exception as e:
        st.error(f"Error generating QA pairs: {str(e)}")
        return pd.DataFrame()

def create_jsonl_content(df: pd.DataFrame) -> str:
    jsonl_content = []
    for _, row in df.iterrows():
        entry = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": row['Question']},
                {"role": "assistant", "content": row['Answer']}
            ],
            "context": row['Context']
        }
        jsonl_content.append(json.dumps(entry, ensure_ascii=False))
    return '\n'.join(jsonl_content)

def main():
    st.title("LLM Dataset Generator")
    st.write("Generate QA pairs from PDF documents using language models.")

    with st.sidebar:
        api_key = st.text_input("Enter Together AI API Key", type="password")
        if api_key and len(api_key) < 20:
            st.warning("Please enter a valid API key")

        model_name = st.selectbox("Select Model", list(MODEL_CONFIGS.keys()), index=0)
        language = st.selectbox("Select Output Language", ["English", "French"], index=0)
        num_pairs = st.number_input("Number of QA Pairs", min_value=1, max_value=50, value=5)
        train_size = st.slider("Training Set Size (%)", min_value=50, max_value=90, value=80, step=5)
        output_format = st.selectbox("Output Format", ["CSV", "JSONL"])

    init_session_state()
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    current_upload_state = uploaded_file is not None
    if current_upload_state != st.session_state.previous_upload_state:
        if not current_upload_state:
            reset_session_state()
        st.session_state.previous_upload_state = current_upload_state

    if uploaded_file is not None and api_key:
        with st.spinner("Processing PDF..."):
            text = parse_pdf(uploaded_file)
            if text:
                st.success("PDF processed successfully!")
                with st.expander("Preview extracted text"):
                    st.text(text[:500] + "...")

                if st.button("Generate QA Pairs"):
                    with st.spinner(f"Generating QA pairs using {model_name}..."):
                        selected_model = MODEL_CONFIGS[model_name]
                        df = generate_qa_pairs(text, api_key, selected_model, num_pairs, language)
                        
                        if not df.empty:
                            train_df, val_df = train_test_split(df, train_size=train_size/100, random_state=42)
                            st.session_state.train_df = train_df
                            st.session_state.val_df = val_df
                            st.session_state.generated = True
                            st.success("QA pairs generated successfully!")

        if st.session_state.generated:
            st.subheader("Generation Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pairs", len(df))
            with col2:
                st.metric("Training Pairs", len(st.session_state.train_df))
            with col3:
                st.metric("Validation Pairs", len(st.session_state.val_df))

            st.subheader("Generated Datasets")
            tab1, tab2 = st.tabs(["Training Set", "Validation Set"])
            
            with tab1:
                st.dataframe(st.session_state.train_df, use_container_width=True)
                if output_format == "CSV":
                    train_csv = st.session_state.train_df.to_csv(index=False)
                    st.download_button("Download Training Set (CSV)", train_csv, "train_qa_pairs.csv", "text/csv")
                else:
                    train_jsonl = create_jsonl_content(st.session_state.train_df)
                    st.download_button("Download Training Set (JSONL)", train_jsonl, "train_qa_pairs.jsonl", "application/jsonl")
            
            with tab2:
                st.dataframe(st.session_state.val_df, use_container_width=True)
                if output_format == "CSV":
                    val_csv = st.session_state.val_df.to_csv(index=False)
                    st.download_button("Download Validation Set (CSV)", val_csv, "val_qa_pairs.csv", "text/csv")
                else:
                    val_jsonl = create_jsonl_content(st.session_state.val_df)
                    st.download_button("Download Validation Set (JSONL)", val_jsonl, "val_qa_pairs.jsonl", "application/jsonl")

if __name__ == "__main__":
    st.set_page_config(page_title="LLM Dataset Generator", page_icon="📚", layout="wide")
    main()