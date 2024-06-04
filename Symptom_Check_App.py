import streamlit as st
from utils import get_data, diseases, metrics_table
from transformers import pipeline
import pandas as pd 
import io
import json
from openai import OpenAI
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


# Load model
model_name = "AlperenEvci/bert-symptom-diagnosis"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
import google.generativeai as genai

def get_data(file:str):
    with open(file) as files:
        data = json.load(files)
    return data
#Bu Python fonksiyonu, bir JSON dosyasından veri okur ve yüklenen veriyi döndürür. İşlevin parametreleri ve dönüş değeri aşağıdaki gibidir:
#file (str): Okunacak JSON dosyasının yolu.
#Dönen değer: Belirtilen JSON dosyasından yüklenen veri (bir sözlük).
#Fonksiyonun çalışma mantığı şu şekildedir:
#with open(file) as files: satırı, belirtilen dosyayı açar ve bu dosyayı files adlı bir dosya nesnesine atar.
#json.load(files) satırı, dosya nesnesini kullanarak JSON verisini yükler ve bir Python sözlüğü olarak döndürür.
#Fonksiyon, yüklenen veriyi data adlı bir değişkende saklar ve bu veriyi döndürür.
 

def callGemınıAI(text,prompt):
    genai.configure(api_key=text)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# Define functions
def get_predictions(text):
    # Metni tokenize et
    inputs = tokenizer(text, return_tensors="tf", truncation=True, max_length=512)
    outputs = model(inputs) # Modeli kullanarak tahminleri al
    probabilities = tf.nn.softmax(outputs.logits, axis=-1) # Olasılıkları hesapla
    predictions = [] # Tahminleri saklamak için boş bir liste oluştur  
    for idx, prob in enumerate(probabilities[0]): # Her bir tahmini döngüyle gez     
        label = model.config.id2label[idx]# Etiketi al    
        predictions.append((label, prob.numpy())) # Tahmini ve olasılığı listeye ekle
    return predictions# Tahminleri döndür

def analyze_probabilities(predictions):
    # Tahminlerdeki olasılıkları bir listede topluyoruz
    probabilities = [prob for _, prob in predictions]
    # En yüksek olasılığı buluyoruz
    max_prob = max(probabilities)
    # En yüksek olasılığa sahip etiketi buluyoruz
    high_prob_label = next(label for label, prob in predictions if prob == max_prob)
    # En yüksek olasılığa sahip etiketi ve bu olasılığı döndürüyoruz
    return high_prob_label, max_prob


def main():
    # import symptoms data
    symptoms = get_data('symptoms_eng.json')

    # Create sidebar and pages content
    tabs = ["Home", "About Us", "Model Details & Evaluations"]
    st.sidebar.header("Welcome to the Symptom checker app ! This app provide diagnosis based on your symptoms. Feel free to try it out !")
    st.sidebar.divider() 
    active_tab = st.sidebar.radio("Select Tab", tabs)
    if active_tab == "Home":
        st.header("Symptom Checker and Diagnosis App", divider = 'violet')
 
        st.subheader("👩‍⚕️ Enter your symptoms:")
        st.subheader("get your diagnosis and useful advices")
  
        input_text = ""
        for category, symptoms in symptoms.items():
            st.sidebar.write(f"### {category}")
            category_symptoms = st.sidebar.multiselect(f"Select Symptoms in {category}", [symptom[0] for symptom in symptoms])
            if category_symptoms:
                input_text += f""
                for selected_symptom in category_symptoms:
                    symptom_description = next((symptom[1] for symptom in symptoms if symptom[0] == selected_symptom), "")
                    input_text += f"{symptom_description}\n"
   
        # Display the updated input_text
        manual_input=st.text_area("Symptoms", value=input_text, height=200)

        st.warning("If you want to know more, please enter your api key and click on submit: ")
        with st.expander("Click here to enter your api"):
            api=st.text_input("API KEY", value="",type='password')

        # Button to submit and get the predicted labela
        if st.button("Submit"):
            predictions = get_predictions(manual_input)
            predicted_label,probability=analyze_probabilities(predictions)

            # Condition to display only high probability deseases
            if probability > 0.6:
                st.success(f"Based on your symptoms, there's a {100 * probability:.2f}% probability that you might have {predicted_label}.")
                if api:
     
                    # Try using the provided API key to call GPT-3
                    prompt = f"Lütfen bu hastalığın kısa bir açıklamasıyla birlikte {predicted_label} için ilaçların bir listesini sağlayın."
                    response = callGemınıAI(api,prompt)

                    try :
                        gpt3_response = callGemınıAI(api,prompt)
                        # Display the GPT-3 response if successful
                        st.header(f'Information about the **{predicted_label}**')
                        st.info('Lütfen unutmayın: Bu bilgiler yapay zeka tarafından oluşturulmuştur ve profesyonel tıbbi tavsiyelerin yerine geçmez.')
                        st.write(gpt3_response)
  
                    except:
                    # Handle case where API call fails
                        st.warning("Unable to retrieve information using the provided API key. Please try another API key if available.")
                else:
                    st.write('If you want to know more, please enter your api key')

            if probability > 0.4 and probability < 0.6:
                st.warning(f"Based on your symptoms, there's a {100 * probability:.2f}% probability that you might have {predicted_label}.")
                if api:
     
                    # Try using the provided API key to call GPT-3
                    prompt = f"Lütfen bu hastalığın kısa bir açıklamasıyla birlikte {predicted_label} için ilaçların bir listesini sağlayın."
                    response = callGemınıAI(api,prompt)

                    try :
                        gpt3_response = callGemınıAI(api,prompt)
                        # Display the GPT-3 response if successful
                        st.header(f'Information about the **{predicted_label}**')
                        st.info('Lütfen unutmayın: Bu bilgiler yapay zeka tarafından oluşturulmuştur ve profesyonel tıbbi tavsiyelerin yerine geçmez.')
                        st.write(gpt3_response)
  
                    except:
                    # Handle case where API call fails
                        st.warning("Unable to retrieve information using the provided API key. Please try another API key if available.")
                else:
                    st.write('If you want to know more, please enter your api key')

            else:
                st.warning("The symptoms you've described do not strongly indicate any of the 22 diseases in our database with a high probability. It's recommended to consult a healthcare professional for a more accurate diagnosis.")
                # Expander to show the list of diseases
                with st.expander("Click here to view the list of diseases"):
                    for disease in diseases:
                        st.write(disease)

    elif active_tab == "Hakkımızda":
            st.title("AkarE")
                    
            st.markdown("**[Alperen EVCİ](https://www.linkedin.com/in/alperen-evci/)**")
            st.markdown("**[Emir SELENGİL](https://www.linkedin.com/in/emir-selengil/)**")
            st.markdown("**[Akın BEKTAŞ](https://www.linkedin.com/in/akin-bektas/)**")
            st.markdown(" We are a dynamic duo of data scientists collaborating to enhance our skills and stay at the forefront of the latest developments. With backgrounds in science and experience working with health data, we bring a unique blend of expertise to our data science projects. Our shared passion and commitment drive us to showcase and elevate our capabilities through innovative and impactful initiatives. Join us on this journey of continuous improvement and exploration in the world of data science. ")
            st.markdown(" ")

    elif active_tab == "Model Details & Evaluations":
        st.subheader("Model Overview:")
        st.write("This model is a fine-tuned adaptation of the bert-base-cased architecture, specifically designed for text classification tasks associated with diagnosing diseases based on symptoms. The primary goal is to scrutinize natural language symptom descriptions and accurately predict one of 22 potential diagnoses.")
        st.subheader("Dataset Information:")
        st.write("The model was trained on the Gretel/symptom_to_diagnosis dataset, which consists of 1,065 symptom descriptions in English, each labeled with one of the 22 possible diagnoses. This dataset focuses on detailed, fine-grained, single-domain diagnosis, making it suitable for tasks requiring nuanced symptom classification. For those interested in utilizing the model, the Symptom Checker and Diagnosis App, or the Inference API, are accessible at [https://huggingface.co/AlperenEvci/bert-symptom-diagnosis](https://huggingface.co/AlperenEvci/bert-symptom-diagnosis).")
        st.subheader("Model Performance Metrics:")
        metrics_data = pd.read_csv(io.StringIO(metrics_table), sep="|").dropna()
        st.table(metrics_data)



if __name__ == "__main__":
    main()