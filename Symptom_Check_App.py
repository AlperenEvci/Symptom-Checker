import streamlit as st
from utils import get_data, diseases, metrics_table
from transformers import pipeline
import pandas as pd 
import io
import json
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import google.generativeai as genai

# Load model
#model_name = "AlperenEvci/bert-symptom-diagnosis"
#bunu aşağıdaki gibi değiştirdik çünkü modeli huggingface'den çekmek yerine localde çalıştırıyoruz
#kodu atarken dikkat et

model_name = "bert-symptom-diagnosis"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# def get_data(file:str):
#     with open(file, encoding='utf-8') as files:
#         data = json.load(files)
#     return data
# #Bu Python fonksiyonu, bir JSON dosyasından veri okur ve yüklenen veriyi döndürür. İşlevin parametreleri ve dönüş değeri aşağıdaki gibidir:
# #file (str): Okunacak JSON dosyasının yolu.
# #Dönen değer: Belirtilen JSON dosyasından yüklenen veri (bir sözlük).
# #Fonksiyonun çalışma mantığı şu şekildedir:
# #with open(file) as files: satırı, belirtilen dosyayı açar ve bu dosyayı files adlı bir dosya nesnesine atar.
# #json.load(files) satırı, dosya nesnesini kullanarak JSON verisini yükler ve bir Python sözlüğü olarak döndürür.
# #Fonksiyon, yüklenen veriyi data adlı bir değişkende saklar ve bu veriyi döndürür.
 

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
    #I want to add exit button()
    # Create sidebar and pages content
    tabs = ["Giriş", "Hakkımızda", "Model Detayları ve Değerlendirmeleri"]
    st.sidebar.header("Symptom checker uygulamasına hoş geldiniz! Bu uygulama, semptomlarınıza dayalı teşhis sağlar. Rahatça deneyebilirsiniz!")
    st.sidebar.divider() 
    active_tab = st.sidebar.radio("Select Tab", tabs) 
    if active_tab == "Giriş":
        st.header("Semptom Kontrol ve Teşhis Uygulaması", divider = 'violet')
 
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

        st.warning("Daha fazla bilgi almak istiyorsanız lütfen API anahtarınızı girin ve Göndere tıklayın:")
        with st.expander("API'nizi girmek için buraya tıklayın:"):
            api=st.text_input("API KEY", value="",type='password')

        # Button to submit and get the predicted labela
        if st.button("Gönder"):
            predictions = get_predictions(manual_input)
            predicted_label,probability=analyze_probabilities(predictions)

            # Condition to display only high probability deseases
            if probability > 0.6:
                st.success(f"Semptomlarınıza dayanarak, {100 * probability:.2f}% olasılıkla {predicted_label} olduğunuzu düşünüyoruz.")
                if api:
     
                    # Try using the provided API key to call GPT-3
                    prompt = f"Lütfen bu hastalığın kısa bir açıklamasıyla birlikte {predicted_label} için ilaçların bir listesini sağlayın."
                    response = callGemınıAI(api,prompt)

                    try :
                        prompts = callGemınıAI(api,prompt)
                        # Display the GPT-3 response if successful
                        st.header(f'**{predicted_label}** hakkında bilgi')
                        st.info('Lütfen unutmayın: Bu bilgiler yapay zeka tarafından oluşturulmuştur ve profesyonel tıbbi tavsiyelerin yerine geçmez.')
                        st.write(prompts)
  
                    except:
                    # Handle case where API call fails
                        st.warning("Unable to retrieve information using the provided API key. Please try another API key if available.")
                else:
                    st.write('Daha fazla bilgi almak istiyorsanız lütfen API anahtarınızı girin ve Göndere tıklayın')

            elif probability > 0.4 and probability < 0.6:
                st.warning(f"Semptomlarınıza dayanarak, {100 * probability:.2f}% olasılıkla {predicted_label} olduğunuzu düşünüyoruz.")
                if api:
     
                    # Try using the provided API key to call GPT-3
                    prompt = f"Lütfen bu hastalığın kısa bir açıklamasıyla birlikte {predicted_label} için ilaçların bir listesini sağlayın."
                    response = callGemınıAI(api,prompt)

                    try :
                        prompts = callGemınıAI(api,prompt)
                        # Display the GPT-3 response if successful
                        st.header(f'**{predicted_label}** hakkında bilgi')
                        st.info('Lütfen unutmayın: Bu bilgiler yapay zeka tarafından oluşturulmuştur ve profesyonel tıbbi tavsiyelerin yerine geçmez.')
                        st.write(prompts)
  
                    except:
                    # Handle case where API call fails
                        st.warning("Sağlanan API anahtarıyla bilgi alınamıyor. Lütfen başka bir API anahtarı deneyin, eğer mevcutsa.")
                else:
                    st.write('Daha fazla bilgi almak istiyorsanız lütfen API anahtarınızı girin ve Göndere tıklayın')

            else:
                st.warning("Tarif ettiğiniz semptomlar, veritabanımızdaki 22 hastalıktan herhangi birini yüksek bir olasılıkla belirtmiyor. Daha doğru bir teşhis için bir sağlık profesyoneline danışmanız önerilir.")
                # Expander to show the list of diseases
                with st.expander("Hastalık listesini görmek için buraya tıklayın:"):
                    for disease in diseases:
                        st.write(disease)

    elif active_tab == "Hakkımızda":
            st.title("AkarE")
                    
            st.markdown("**[Alperen EVCİ](https://www.linkedin.com/in/alperen-evci/)**")
            st.markdown("**[Emir SELENGİL](https://www.linkedin.com/in/emir-selengil/)**")
            st.markdown("**[Akın BEKTAŞ](https://www.linkedin.com/in/akin-bektas/)**")
            st.markdown(" Merhaba! Biz, veri bilimine yeni adım atan üç kişilik bir ekip olarak buradayız. Sürekli olarak yeni şeyler öğrenmeye ve alanın gelişmelerini takip etmeye çalışıyoruz. Farklı bilim geçmişlerimizle, projelerimize çeşitlilik katıyoruz. Ortak bir tutkumuz var ve bu tutku, bizi yaratıcı ve etkili projelere yönlendiriyor. Eğer bizimle birlikte veri bilimi dünyasına adım atmak isterseniz, sizi aramızda görmekten mutluluk duyarız! ")
            st.markdown(" ")

    elif active_tab == "Model Detayları ve Değerlendirmeleri":
        st.subheader("Model Overview:")
        st.write("Bu model, bert-base-cased mimarisinin fine-tuned bir uyarlamasıdır ve özellikle semptomlara dayalı hastalıkları teşhis etme ile ilgili metin sınıflandırma görevleri için tasarlanmıştır. Temel amaç, doğal dildeki semptom açıklamalarını detaylı bir şekilde incelemek ve 22 potansiyel teşhisi doğru bir şekilde tahmin etmektir.")
        st.subheader("Dataset Information:")
        st.write("Model, 1.065 semptom açıklamasından oluşan İngilizce Gretel/symptom_to_diagnosis veri kümesi üzerinde eğitildi. Her bir semptom açıklaması, 22 olası teşhisten biriyle etiketlenmiştir. Bu veri kümesi, ayrıntılı, ince detaylı ve tek bir alan olan teşhislere odaklanarak, nüanslı semptom sınıflandırması gerektiren görevler için uygun hale getirilmiştir. Modeli kullanmak isteyenler için, Semptom Kontrol ve Teşhis Uygulaması veya Çıkarım API'si, [https://huggingface.co/AlperenEvci/bert-symptom-diagnosis](https://huggingface.co/AlperenEvci/bert-symptom-diagnosis). adresinde erişilebilir.")
        st.subheader("Model Performance Metrics:")
        metrics_data = pd.read_csv(io.StringIO(metrics_table), sep="|").dropna()
        st.table(metrics_data)





if __name__ == "__main__":
    main()