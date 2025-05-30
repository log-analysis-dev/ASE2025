import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

# from pymongo import MongoClient
import re

# mongo_uri = "mongodb+srv://username:pw@llm.uqk1w.mongodb.net/?retryWrites=true&w=majority&appName=llm&tlsAllowInvalidCertificates=true"
# db_name = "ai"
# collection_name = "model_parsing_history"

# # Connect to MongoDB
# try:
#   client = MongoClient(mongo_uri)
#   client.admin.command("ping")
#   print("MongoDB connection successful!")
# except Exception as e:
#   print(f"Error connecting to MongoDB: {e}")

# # Access db and connection
# db = client[db_name]
# collection = db[collection_name]


# Function to process the 'answer_by_model' column
def process_answer_by_model(answer_text):
    """
    Extract the content after 'Let's think it step by step.' in the LLM-generated response.
    """
    # match = re.search(r"### Answer:\s*(.+)", answer_text, re.DOTALL)
    match = re.search(r"Let's think it step by step.\s*(.+)", answer_text, re.DOTALL)
    return match.group(1).strip() if match else None



def load_model(model_name):
    try:
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_buffers=True,
            token=HF_TOKEN
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
            model_max_length=1024  
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        print(f"Successfully loaded model: {model_name}")
        return model, tokenizer, None
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None, None, None


def generate_answer(model, tokenizer, device, question_text, max_new_tokens=1024):
    try:
        inputs = tokenizer(
            question_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during generation: {e}")
        return "Generation failed"


def process_parsing(file_path, models, output_dir, n):
    questions = pd.read_excel(file_path)
    total_questions = len(questions)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    for model_name in models:
        for iteration in range(1, n + 1):  # Run each model n times
            model, tokenizer, device = load_model(model_name)
            if model is None or tokenizer is None:
                print(f"Skipping model {model_name} due to load failure.\n")
                continue  # Skip to the next model

            results = []

            print(f"Processing file: {file_path} with model: {model_name} (Run {iteration})")
            # Step 1: Load few-shot example set (selected via DPP and annotated manually)
            # The file must contain 'log' (raw log) and 'parsed' (structured output) columns
            few_shot_df = pd.read_excel("few_shot_examples.xlsx")  # Make sure this file exists

            # Preprocessing: Fit a vectorizer to tokenize logs (space-separated words)
            vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b\w+\b').fit(few_shot_df['log'])

            # Vectorized representation of few-shot logs
            few_shot_vectors = vectorizer.transform(few_shot_df['log'])

            # Iterate over each raw log entry to be parsed
            for idx, row in questions.iterrows():
                log_id = row.get('log_id', idx)
                raw_log = row['raw_log']

                # Step 2: Vectorize the current raw log
                raw_vector = vectorizer.transform([raw_log])

                # Step 3: Compute Jaccard similarity between current log and each few-shot example
                jaccard_scores = []
                for i in range(few_shot_vectors.shape[0]):
                    a = raw_vector.toarray()[0]
                    b = few_shot_vectors[i].toarray()[0]
                    intersection = np.sum(np.logical_and(a, b))
                    union = np.sum(np.logical_or(a, b))
                    score = intersection / union if union != 0 else 0
                    jaccard_scores.append(score)

                # Step 4: Select top-5 most similar examples using kNN
                top_indices = np.argsort(jaccard_scores)[-5:][::-1]
                selected_examples = few_shot_df.iloc[top_indices]

                # Step 5: Construct the ICL prompt with instruction + 5 few-shot examples + current log
                instruction = "Please parse the following log message into structured components such as timestamp, log level, IP, user, or message.\n\n"
                example_prompt = ""
                for _, ex in selected_examples.iterrows():
                    example_prompt += f"Log: {ex['log']}\nParsed: {ex['parsed']}\n\n"

                parsing_prompt = (
                        instruction +
                        example_prompt +
                        f"Log: {raw_log}\nParsed:"
                )

                print(f"Processed Log {idx + 1}/{len(questions)}")
                print(f"Log ID: {log_id}")
                print(f"Prompt:\n{parsing_prompt}")

                # Model inference (you should have defined generate_answer and process_answer_by_model)
                generated_answer = generate_answer(model, tokenizer, device, parsing_prompt, max_new_tokens=1024)
                answer_by_model_processed = process_answer_by_model(generated_answer)
                print(generated_answer)

                # Store results
                result = {
                    'base_model': model_name,
                    'iteration': iteration,
                    'parsingpaper': file_name,
                    'log_id': log_id,
                    'raw_log': raw_log,
                    'instruction': parsing_prompt,
                    'answer_by_model': generated_answer,
                    'answer_by_model_processed': answer_by_model_processed,
                    'prompt': 'few-shot-knn',
                    'isAssessed': bool(False),
                    'created_at': pd.Timestamp.now().isoformat(),
                    'updated_at': pd.Timestamp.now().isoformat()
                }

                # Uncomment if collecting results in a list
                # results.append(result)

                # print(result)

                #Save individual question result as JSON
                json_file_path = os.path.join(output_dir, f"{file_name}_Q{qId}_run{iteration}.json")
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    # result.pop('_id', None)
                    # result['created_at'] = result['created_at'].isoformat()
                    # result['updated_at'] = result['updated_at'].isoformat()

                    json.dump(result, json_file, ensure_ascii=False, indent=4)
                print(f"Saved result to {json_file_path}")

                results.append(result)
                print("----completed this question-----")

            # Save results to Excel
            output_file = os.path.join(output_dir, f"{file_name}_{model_name.replace('/', '_')}_run{iteration}.xlsx")
            pd.DataFrame(results).to_excel(output_file, index=False)
            print(f"Results saved to {output_file}\n")


if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    models = [
        # 'meta-llama/Llama-3.2-1B',
        # 'meta-llama/Llama-3.1-8B',
        'Qwen/Qwen2.5-3B', 
        # 'Qwen/Qwen2.5-7B',
        # 'google/gemma-2-2b', 
        # 'google/gemma-2-9b'
    ]
    n = 5  # Number of times to run each model
    
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(input_dir):
        if file.endswith(".xlsx"):
            file_path = os.path.join(input_dir, file)
            process_parsing(file_path, models, output_dir, n)
    print("All parsing processed.")
