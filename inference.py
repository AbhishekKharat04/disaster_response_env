"""
Disaster Response Environment — Inference Script
Runs baseline agent against all 3 tasks and prints scores.
"""
import requests

BASE_URL = "https://abhishekkharat11-disaster-response-env.hf.space"

def main():
    print("Running inference against all 3 tasks...")
    r = requests.get(f"{BASE_URL}/baseline", timeout=60)
    data = r.json()
    for result in data["results"]:
        print(f"Task {result['task_level']}: {result['task_name']}")
        print(f"  Final score: {result['final_score']:.4f} / 1.0")
    print(f"Overall: {data['overall_score']:.4f} / 1.0")

if __name__ == "__main__":
    main()