from transformers import pipeline

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

classifier = pipeline("sentiment-analysis", 
                       model=model_name)
#classifier.save_pretrained('model-sent-anlysis')

#classifier = pipeline("sentiment-analysis", 
#                      model="model-sent-anlysis")
results = classifier([
    "We are very happy to see you", 
    "Nous somme heureux de vous accueillir",
    "Produit de mauvaise qualité !",
    "! منظر جميل للغاية",
    "! منتجات جيدة للغاية"])

print("")
print("")
print("-"*40)
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
print("-"*40)
print("")
print("")