"""
Disaster Response Coordination Environment — Baseline Agent
Calls the /baseline endpoint which runs all 3 tasks server-side.
Usage: python baseline_agent.py --base_url https://YOUR-SPACE.hf.space
"""
import argparse, requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:7860")
    parser.add_argument("--task_level", type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("DISASTER RESPONSE BASELINE AGENT")
    print(f"{'='*60}")
    print(f"Server: {args.base_url}\n")

    url = f"{args.base_url}/baseline"
    if args.task_level > 0:
        url += f"?task_level={args.task_level}"

    r = requests.get(url, timeout=60)
    data = r.json()

    for result in data["results"]:
        print(f"Task {result['task_level']}: {result['task_name']}")
        print(f"  Steps:       {result['steps']}")
        print(f"  Step scores: {result['step_scores']}")
        print(f"  Final score: {result['final_score']:.4f} / 1.0")
        print()

    if "overall_score" in data:
        print(f"OVERALL SCORE: {data['overall_score']:.4f} / 1.0")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()