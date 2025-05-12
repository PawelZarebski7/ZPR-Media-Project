import streamlit as st
import boto3
import json
import uuid
import time
import os
from PIL import Image
import io
import base64

# Konfiguracja AWS
def get_aws_credentials():
    # Odczytanie danych z secrets.toml - nowa struktura
    return {
        'aws_access_key_id': st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
        'aws_secret_access_key': st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
        'region_name': st.secrets["aws"]["AWS_REGION"]
    }

# Inicjalizacja klientów AWS
@st.cache_resource
def init_aws_clients():
    creds = get_aws_credentials()
    s3 = boto3.client('s3', **creds)
    rekognition = boto3.client('rekognition', **creds)
    bedrock = boto3.client('bedrock-runtime', **creds)
    dynamodb = boto3.resource('dynamodb', **creds)
    
    return {
        's3': s3,
        'rekognition': rekognition,
        'bedrock': bedrock,
        'dynamodb': dynamodb
    }

# Funkcja do przesyłania plików do S3
def upload_to_s3(file_bytes, bucket_name):
    clients = init_aws_clients()
    s3 = clients['s3']
    
    file_key = f"uploads/{uuid.uuid4()}.jpg"
    s3.put_object(
        Bucket=bucket_name,
        Key=file_key,
        Body=file_bytes,
        ContentType='image/jpeg'
    )
    
    return file_key

# Funkcja do analizy obrazu z Amazon Rekognition
def analyze_image(bucket, file_key):
    clients = init_aws_clients()
    rekognition = clients['rekognition']
    
    # Detekcja obiektów
    detect_response = rekognition.detect_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': file_key}},
        MaxLabels=20,
        MinConfidence=70
    )
    
    # Detekcja tekstu
    text_response = rekognition.detect_text(
        Image={'S3Object': {'Bucket': bucket, 'Name': file_key}}
    )
    
    # Analiza twarzy
    face_response = rekognition.detect_faces(
        Image={'S3Object': {'Bucket': bucket, 'Name': file_key}},
        Attributes=['ALL']
    )
    
    # Kompilacja wyników
    analysis = {
        'labels': detect_response['Labels'],
        'text': [t['DetectedText'] for t in text_response.get('TextDetections', []) if t['Type'] == 'LINE'],
        'faces': face_response.get('FaceDetails', [])
    }
    
    return analysis

# Funkcja do generowania opisu i tagów z LLM
def generate_description_and_tags(image_analysis):
    clients = init_aws_clients()
    bedrock = clients['bedrock']
    
    # Przygotowanie danych dla LLM
    detected_objects = [f"{label['Name']} (pewność: {label['Confidence']:.1f}%)" 
                      for label in image_analysis['labels']]
    
    detected_text = image_analysis['text']
    
    face_details = []
    for face in image_analysis['faces']:
        emotions = []
        if 'Emotions' in face:
            emotions = [e['Type'] for e in face['Emotions'] if e['Confidence'] > 50]
        
        age_range = f"{face['AgeRange']['Low']}-{face['AgeRange']['High']}"
        gender = face['Gender']['Value']
        face_details.append(f"{gender}, wiek {age_range}, emocje: {', '.join(emotions)}")
    
    # Budowanie promptu dla LLM
    prompt = f"""
    Na podstawie analizy zdjęcia wykryto następujące elementy:
    
    Obiekty: {', '.join(detected_objects)}
    
    {"Tekst na zdjęciu: " + ', '.join(detected_text) if detected_text else ""}
    
    {"Twarze: " + ', '.join(face_details) if face_details else ""}
    
    Na podstawie powyższych danych, proszę wygeneruj:
    1. Szczegółowy opis zdjęcia (3-5 zdań)
    2. Listę 5-10 odpowiednich tagów dla tego zdjęcia
    
    Odpowiedź sformatuj w JSON:
    {{
        "description": "Opis zdjęcia...",
        "tags": ["tag1", "tag2", "tag3", ...]
    }}
    """
    
    try:
        # Wywołanie Amazon Bedrock (Claude) - zaktualizowane ID profilu inferencyjnego
        response = bedrock.invoke_model(
            modelId='eu.anthropic.claude-3-7-sonnet-20250219-v1:0',  # Profil inferencyjny Claude 3.7 Sonnet
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1000,
                'temperature': 0.7,
                'messages': [{'role': 'user', 'content': prompt}]
            })
        )
        
        response_body = json.loads(response['body'].read())
        llm_output = response_body['content'][0]['text']
        
        # Wyodrębnienie JSON z odpowiedzi
        start_index = llm_output.find('{')
        end_index = llm_output.rfind('}') + 1
        json_str = llm_output[start_index:end_index]
        
        # Parsowanie JSON
        result = json.loads(json_str)
        return result
    
    except Exception as e:
        st.error(f"Błąd przy generowaniu opisu: {str(e)}")
        return {
            'description': 'Nie udało się wygenerować opisu.',
            'tags': []
        }

# Uproszona funkcja do obsługi pytań FAQ bez użycia embeddings
def answer_question(question):
    # Przykładowe FAQ
    example_faqs = [
        {
            'question': 'Jak działa ten asystent?',
            'answer': 'Asystent analizuje przesłane zdjęcia używając AI, generuje opisy i tagi, oraz odpowiada na pytania z FAQ.'
        },
        {
            'question': 'Czy moje zdjęcia są przechowywane?',
            'answer': 'Zdjęcia są przechowywane w bezpieczny sposób w chmurze AWS, z kontrolą dostępu.'
        },
        {
            'question': 'Jakie rodzaje zdjęć mogę analizować?',
            'answer': 'Możesz analizować różne typy zdjęć, ale system działa najlepiej ze zdjęciami osób, zwierząt, krajobrazów i przedmiotów.'
        }
    ]
    
    # Znalezienie najbardziej podobnego pytania
    best_match = None
    highest_score = 0
    
    for faq in example_faqs:
        # Proste porównanie słów
        faq_words = set(faq['question'].lower().split())
        question_words = set(question.lower().split())
        common_words = faq_words.intersection(question_words)
        
        if len(faq_words) > 0:
            score = len(common_words) / len(faq_words)
            if score > highest_score:
                highest_score = score
                best_match = faq
    
    if highest_score > 0.5:  # Próg podobieństwa
        return {
            'matched_question': best_match['question'],
            'answer': best_match['answer'],
            'similarity': highest_score
        }
    else:
        # Generowanie odpowiedzi za pomocą Claude
        clients = init_aws_clients()
        bedrock = clients['bedrock']
        
        prompt = f"""
        Użytkownik zadał pytanie: "{question}"
        
        Oto pytania i odpowiedzi z naszego FAQ:
        
        Q: Jak działa ten asystent?
        A: Asystent analizuje przesłane zdjęcia używając AI, generuje opisy i tagi, oraz odpowiada na pytania z FAQ.
        
        Q: Czy moje zdjęcia są przechowywane?
        A: Zdjęcia są przechowywane w bezpieczny sposób w chmurze AWS, z kontrolą dostępu.
        
        Q: Jakie rodzaje zdjęć mogę analizować?
        A: Możesz analizować różne typy zdjęć, ale system działa najlepiej ze zdjęciami osób, zwierząt, krajobrazów i przedmiotów.
        
        Biorąc pod uwagę te informacje, udziel najlepszej możliwej odpowiedzi na pytanie użytkownika.
        Jeśli pytanie nie jest związane z dostępnymi informacjami, powiedz, że nie masz wystarczających danych aby odpowiedzieć.
        """
        
        try:
            llm_response = bedrock.invoke_model(
                modelId='eu.anthropic.claude-3-7-sonnet-20250219-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 1000,
                    'temperature': 0.7,
                    'messages': [{'role': 'user', 'content': prompt}]
                })
            )
            
            llm_body = json.loads(llm_response['body'].read())
            generated_answer = llm_body['content'][0]['text']
            
            return {
                'generated_answer': generated_answer,
                'relevant_faqs': [faq['question'] for faq in example_faqs]
            }
        except Exception as e:
            st.error(f"Błąd przy generowaniu odpowiedzi: {str(e)}")
            return {
                'error': f"Nie udało się przetworzyć pytania: {str(e)}"
            }

# Zapisanie danych w DynamoDB
def save_to_dynamodb(bucket, key, result):
    clients = init_aws_clients()
    dynamodb = clients['dynamodb']
    
    try:
        table = dynamodb.Table('ImageDescriptions')
        
        # Sprawdź czy tabela istnieje
        try:
            table.table_status
        except Exception as e:
            st.warning(f"Problem z tabelą DynamoDB: {str(e)}")
            return False
            
        item = {
            'image_id': key,
            's3_bucket': bucket,
            's3_key': key,
            'description': result['description'],
            'tags': result['tags'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        table.put_item(Item=item)
        return True
    except Exception as e:
        st.error(f"Błąd przy zapisie do DynamoDB: {str(e)}")
        return False

# Główny interfejs Streamlit
def main():
    st.set_page_config(
        page_title="Asystent Opisywania i Tagowania Zdjęć",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Asystent Opisywania i Tagowania Zdjęć")
    
    # Inicjalizacja stanu sesji
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'image_description' not in st.session_state:
        st.session_state.image_description = None
    if 'image_tags' not in st.session_state:
        st.session_state.image_tags = None
    if 'faq_answer' not in st.session_state:
        st.session_state.faq_answer = None
    
    # Tworzenie dwóch głównych zakładek
    tab1, tab2 = st.tabs(["Analiza Zdjęć", "FAQ"])
    
    # Zakładka Analiza Zdjęć
    with tab1:
        st.header("Prześlij zdjęcie do analizy")
        
        uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Wyświetlanie przesłanego zdjęcia
            image = Image.open(uploaded_file)
            st.image(image, caption="Przesłane zdjęcie", use_container_width=True)
            st.session_state.uploaded_image = image
            
            # Przycisk do analizy
            if st.button("Analizuj zdjęcie"):
                with st.spinner("Analizuję zdjęcie..."):
                    # Konwersja obrazu do bytesIO dla AWS
                    buf = io.BytesIO()
                    
                    # Konwersja obrazu z RGBA do RGB, jeśli jest to konieczne
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    
                    image.save(buf, format='JPEG')
                    image_bytes = buf.getvalue()
                    
                    # Bucket S3 - pobierz z konfiguracji
                    bucket_name = st.secrets["aws"]["S3_BUCKET_NAME"]
                    
                    try:
                        # Przesłanie do S3
                        file_key = upload_to_s3(image_bytes, bucket_name)
                            
                        # Analiza obrazu
                        image_analysis = analyze_image(bucket_name, file_key)
                        
                        # Generowanie opisu i tagów
                        result = generate_description_and_tags(image_analysis)
                        
                        # Zapis do DynamoDB
                        save_to_dynamodb(bucket_name, file_key, result)
                        
                        # Zapisanie wyników w stanie sesji
                        st.session_state.image_description = result['description']
                        st.session_state.image_tags = result['tags']
                        
                    except Exception as e:
                        st.error(f"Wystąpił błąd: {str(e)}")
            
            # Wyświetlanie wyników analizy
            if st.session_state.image_description:
                st.subheader("📝 Opis zdjęcia:")
                st.write(st.session_state.image_description)
                
                st.subheader("🏷️ Tagi:")
                tags_html = ' '.join([f'<span style="background-color: #f0f2f6; padding: 5px 10px; border-radius: 20px; margin-right: 8px; font-size: 0.9em;">#{tag}</span>' for tag in st.session_state.image_tags])
                st.markdown(f'<div style="line-height: 2.5;">{tags_html}</div>', unsafe_allow_html=True)
    
    # Zakładka FAQ
    with tab2:
        st.header("Zadaj pytanie dotyczące systemu")
        
        # Pole do wprowadzania pytania
        user_question = st.text_input("Twoje pytanie:")
        
        if st.button("Zadaj pytanie") and user_question:
            with st.spinner("Szukam odpowiedzi..."):
                try:
                    answer_data = answer_question(user_question)
                    st.session_state.faq_answer = answer_data
                except Exception as e:
                    st.error(f"Wystąpił błąd: {str(e)}")
        
        # Wyświetlanie odpowiedzi na pytanie
        if st.session_state.faq_answer:
            st.subheader("Odpowiedź:")
            
            if 'matched_question' in st.session_state.faq_answer:
                st.info(f"Znaleziono podobne pytanie: \"{st.session_state.faq_answer['matched_question']}\"")
                st.write(st.session_state.faq_answer['answer'])
            elif 'generated_answer' in st.session_state.faq_answer:
                st.write(st.session_state.faq_answer['generated_answer'])
                
                if 'relevant_faqs' in st.session_state.faq_answer and st.session_state.faq_answer['relevant_faqs']:
                    st.subheader("Powiązane pytania z FAQ:")
                    for q in st.session_state.faq_answer['relevant_faqs']:
                        st.markdown(f"- {q}")
    
    # Stopka
    st.markdown("---")
    st.markdown("📊 Asystent opisywania i tagowania zdjęć | Projekt rekrutacyjny")

if __name__ == "__main__":
    main()