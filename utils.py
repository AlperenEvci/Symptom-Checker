import json

def get_data(file:str):
    """
    Reads data from a JSON file and returns the loaded data.

    Parameters:
        file (str): The path to the JSON file to be read.

    Returns:
        dict: The data loaded from the specified JSON file.
    """
    with open(file) as files:
        data = json.load(files)
    return data

diseases = [
    "Allergy", "Arthritis", "Bronchial Asthma", "Cervical Spondylosis", 
    "Chicken Pox", "Common Cold", "Dengue", "Diabetes", "Drug Reaction", 
    "Fungal Infection", "Gastroesophageal Reflux Disease", "Hypertension", 
    "Impetigo", "Jaundice", "Malaria", "Migraine", "Peptic Ulcer Disease", 
    "Pneumonia", "Psoriasis", "Typhoid", "Urinary Tract Infection", 
    "Varicose Veins"
]

metrics_table = """
                 Disease                         | Precision | Recall | F1-Score  
                 Allergy                         | 0.91      | 1.00   | 0.95     
                 Arthritis                       | 1.00      | 1.00   | 1.00    
                 Bronchial Asthma                | 1.00      | 1.00   | 1.00      
                 Cervical Spondylosis            | 0.91      | 1.00   | 0.95     
                 Chicken Pox                     | 1.00      | 1.00   | 1.00     
                 Common Cold                     | 1.00      | 1.00   | 1.00     
                 Dengue                          | 1.00      | 0.90   | 0.95     
                 Diabetes                        | 1.00      | 0.80   | 0.89    
                 Drug Reaction                   | 0.80      | 1.00   | 0.89     
                 Fungal Infection                | 1.00      | 1.00   | 1.00      
                 Gastroesophageal Reflux Disease | 1.00      | 0.90   | 0.95      
                 Hypertension                    | 0.91      | 1.00   | 0.95     
                 Impetigo                        | 1.00      | 1.00   | 1.00     
                 Jaundice                        | 1.00      | 1.00   | 1.00      
                 Malaria                         | 1.00      | 1.00   | 1.00      
                 Migraine                        | 1.00      | 0.90   | 0.95     
                 Peptic Ulcer Disease            | 1.00      | 1.00   | 1.00      
                 Pneumonia                       | 1.00      | 1.00   | 1.00     
                 Psoriasis                       | 1.00      | 0.90   | 0.95     
                 Typhoid                         | 1.00      | 1.00   | 1.00     
                 Urinary Tract Infection         | 0.90      | 1.00   | 0.95     
                 Varicose Veins                  | 1.00      | 1.00   | 1.00     
            """