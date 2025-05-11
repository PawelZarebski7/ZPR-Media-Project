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
    # W środowisku produkcyjnym lepiej używać IAM roles
    return {
        'aws_access_key_id': st.secrets.get("AWS_ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID")),
        'aws_secret_access_key': st.secrets.get("AWS_SECRET_ACCESS_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY")),
        'region_name': st.secrets.get("AWS_REGION", os.environ.get("AWS_REGION", "eu-central-1"))
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
        # Wywołanie Amazon Bedrock (Claude)
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
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

# Funkcja do obsługi pytań FAQ
def answer_question(question):
    clients = init_aws_clients()
    bedrock = clients['bedrock']
    dynamodb = clients['dynamodb']
    
    # Generowanie embeddingu dla pytania użytkownika
    try:
        embed_response = bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=json.dumps({
                'inputText': question,
                'embeddingConfig': {
                    'outputDimension': 1536
                }
            })
        )
        
        embed_body = json.loads(embed_response['body'].read())
        question_embedding = embed_body['embedding']
        
        # Przeszukanie bazy FAQ
        faq_table = dynamodb.Table('FAQ')
        all_items = faq_table.scan()['Items']
        
        # Jeśli baza FAQ jest pusta, użyjemy hardcoded przykładów do demonstracji
        if not all_items:
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
            
            # Symulacja embeddingów dla przykładowych FAQ
            example_faqs_with_embeddings = []
            for faq in example_faqs:
                embed_resp = bedrock.invoke_model(
                    modelId='amazon.titan-embed-text-v1',
                    body=json.dumps({
                        'inputText': faq['question'],
                        'embeddingConfig': {
                            'outputDimension': 1536
                        }
                    })
                )
                
                embed_body = json.loads(embed_resp['body'].read())
                faq['embedding'] = embed_body['embedding']
                example_faqs_with_embeddings.append(faq)
            
            all_items = example_faqs_with_embeddings
        
        # Obliczenie podobieństwa między pytaniem użytkownika a pytaniami z FAQ
        similarities = []
        for item in all_items:
            faq_embedding = item['embedding']
            # Obliczenie podobieństwa kosinusowego
            dot_product = sum(a * b for a, b in zip(question_embedding, faq_embedding))
            magnitude1 = sum(a * a for a in question_embedding) ** 0.5
            magnitude2 = sum(b * b for b in faq_embedding) ** 0.5
            
            if magnitude1 * magnitude2 == 0:
                similarity = 0
            else:
                similarity = dot_product / (magnitude1 * magnitude2)
            
            similarities.append({
                'question': item['question'],
                'answer': item['answer'],
                'similarity': similarity
            })
        
        # Sortowanie po podobieństwie
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        best_match = similarities[0]
        
        # Jeśli podobieństwo jest wystarczająco wysokie, zwracamy bezpośrednio odpowiedź z FAQ
        if best_match['similarity'] > 0.85:
            return {
                'matched_question': best_match['question'],
                'answer': best_match['answer'],
                'similarity': best_match['similarity']
            }
        else:
            # W przeciwnym razie, używamy LLM do generowania odpowiedzi
            relevant_faqs = [f"Q: {s['question']}\nA: {s['answer']}" for s in similarities[:3]]
            context = "\n\n".join(relevant_faqs)
            
            prompt = f"""
            Użytkownik zadał pytanie: "{question}"
            
            Oto najbardziej zbliżone pytania i odpowiedzi z naszego FAQ:
            
            {context}
            
            Biorąc pod uwagę te informacje, udziel najlepszej możliwej odpowiedzi na pytanie użytkownika.
            Jeśli pytanie nie jest związane z dostępnymi informacjami, powiedz, że nie masz wystarczających danych aby odpowiedzieć.
            """
            
            llm_response = bedrock.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
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
                'relevant_faqs': [s['question'] for s in similarities[:3]]
            }
    
    except Exception as e:
        st.error(f"Błąd przy odpowiadaniu na pytanie: {str(e)}")
        return {
            'error': f"Nie udało się przetworzyć pytania: {str(e)}"
        }

# Zapisanie danych w DynamoDB
def save_to_dynamodb(bucket, key, result):
    clients = init_aws_clients()
    dynamodb = clients['dynamodb']
    
    try:
        table = dynamodb.Table('ImageDescriptions')
        
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
            st.image(image, caption="Przesłane zdjęcie", use_column_width=True)
            st.session_state.uploaded_image = image
            
            # Przycisk do analizy
            if st.button("Analizuj zdjęcie"):
                with st.spinner("Analizuję zdjęcie..."):
                    # Konwersja obrazu do bytesIO dla AWS
                    buf = io.BytesIO()
                    image.save(buf, format='JPEG')
                    image_bytes = buf.getvalue()
                    
                    # Bucket S3 - w produkcji użyj rzeczywistego bucketa
                    bucket_name = "my-image-assistant-bucket"
                    
                    try:
                        # Przesłanie do S3 (w trybie demonstracyjnym możemy pominąć)
                        file_key = f"demo-image-{uuid.uuid4()}.jpg"
                        
                        # Jeśli mamy skonfigurowany AWS, przesyłamy rzeczywiście
                        if get_aws_credentials()['aws_access_key_id']:
                            file_key = upload_to_s3(image_bytes, bucket_name)
                            
                        # Analiza obrazu
                        image_analysis = analyze_image(bucket_name, file_key)
                        
                        # Generowanie opisu i tagów
                        result = generate_description_and_tags(image_analysis)
                        
                        # Zapis do DynamoDB (jeśli mamy skonfigurowany AWS)
                        if get_aws_credentials()['aws_access_key_id']:
                            save_to_dynamodb(bucket_name, file_key, result)
                        
                        # Zapisanie wyników w stanie sesji
                        st.session_state.image_description = result['description']
                        st.session_state.image_tags = result['tags']
                        
                    except Exception as e:
                        st.error(f"Wystąpił błąd: {str(e)}")
                        
                        # Dla demonstracji, jeśli AWS nie jest skonfigurowany
                        st.warning("Używam danych demonstracyjnych (AWS nie skonfigurowany)")
                        st.session_state.image_description = "To zdjęcie przedstawia górski krajobraz z jeziorem otoczonym przez sosnowy las. W tle widać ośnieżone szczyty gór, a nad nimi błękitne niebo z kilkoma pierzastymi chmurami. Na pierwszym planie widoczny jest fragment kamienistej plaży."
                        st.session_state.image_tags = ["góry", "jezioro", "las", "krajobraz", "natura", "niebo", "chmury", "skały", "odbicie", "spokój"]
            
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
                    
                    # Dla demonstracji, jeśli AWS nie jest skonfigurowany
                    st.warning("Używam danych demonstracyjnych (AWS nie skonfigurowany)")
                    st.session_state.faq_answer = {
                        'generated_answer': "Ten asystent analizuje przesłane zdjęcia używając sztucznej inteligencji, generuje szczegółowe opisy i tagi, które pomagają w kategoryzacji i wyszukiwaniu. Przesłane zdjęcia są przetwarzane przez usługi AWS Rekognition oraz Amazon Bedrock (Claude) do analizy zawartości i generowania opisów naturalnym językiem."
                    }
        
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