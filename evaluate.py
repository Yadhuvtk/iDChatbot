import time
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from core.pipeline import SOPPipeline

# Configuration
TEST_DATA_PATH = 'data/test_data.csv'  # Must have 'question' and 'answer' columns
REPORT_PATH = 'evaluation_report.csv'
SIMILARITY_THRESHOLD = 0.6  # Score (0-1) to consider an answer "Correct"

def load_evaluation_model():
    """
    Loads a separate, standard model to act as the 'Judge'.
    We use MiniLM because it is fast and good at comparing meaning.
    """
    print("⏳ Loading Evaluation Model (Judge)...")
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def evaluate():
    # 1. Check for Test Data
    try:
        df = pd.read_csv(TEST_DATA_PATH)
        # Ensure columns exist (case insensitive check)
        df.columns = df.columns.str.lower()
        if 'question' not in df.columns or 'answer' not in df.columns:
            print(f"❌ Error: {TEST_DATA_PATH} must have 'question' and 'answer' columns.")
            return
    except FileNotFoundError:
        print(f"❌ Error: Could not find {TEST_DATA_PATH}")
        print("   Please create a CSV with two columns: question, answer")
        return

    # 2. Load Your Chatbot Pipeline
    print("⏳ Loading Chatbot Pipeline...")
    bot = SOPPipeline()
    bot.load()

    # 3. Load Judge Model
    judge_model = load_evaluation_model()

    print(f"\n🚀 Starting Evaluation on {len(df)} questions...\n")
    
    results = []
    correct_count = 0
    total_time = 0

    # 4. Run the Loop
    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = str(row['question'])
        ground_truth = str(row['answer'])

        # --- Ask the Chatbot ---
        start_ts = time.time()
        output = bot.query(question)
        duration = time.time() - start_ts
        total_time += duration

        # Get the bot's response
        # (Handles both your original pipeline and the edited one)
        generated_response = output.get('response', output.get('answer', ''))
        is_rejected = output.get('rejected', False)
        confidence_score = output.get('score', 0.0)

        # --- Judge the Answer ---
        status = "FAIL"
        similarity = 0.0

        if is_rejected:
            # If the bot refused to answer, it's a FAIL (False Negative)
            # unless the ground truth explicitly says "I don't know"
            status = "REJECTED"
            similarity = 0.0
        else:
            # Semantic Similarity Check
            # Encode both the expected answer and the bot's answer
            emb1 = judge_model.encode(ground_truth, convert_to_tensor=True)
            emb2 = judge_model.encode(generated_response, convert_to_tensor=True)
            
            # Calculate Cosine Similarity (0.0 to 1.0)
            similarity = util.cos_sim(emb1, emb2).item()

            if similarity >= SIMILARITY_THRESHOLD:
                status = "PASS"
                correct_count += 1
            else:
                status = "FAIL (Low Accuracy)"

        # Save detailed result
        results.append({
            "question": question,
            "expected_answer": ground_truth,
            "bot_response": generated_response,
            "status": status,
            "similarity_score": round(similarity, 3),
            "pipeline_confidence": round(confidence_score, 3),
            "latency_sec": round(duration, 3),
            "rejected": is_rejected
        })

    # 5. Calculate Final Statistics
    accuracy = (correct_count / len(df)) * 100
    avg_latency = total_time / len(df)

    print("\n" + "="*30)
    print("📊 EVALUATION SUMMARY")
    print("="*30)
    print(f"Total Questions: {len(df)}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print(f"Avg Latency:     {avg_latency:.4f} sec/query")
    print("="*30)

    # 6. Save Report
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_PATH, index=False)
    print(f"\n✅ Detailed report saved to: {REPORT_PATH}")
    print("Check this file to see exactly where the bot failed.")

if __name__ == "__main__":
    evaluate()