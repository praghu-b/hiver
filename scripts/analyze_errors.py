import sys
sys.path.insert(0, ".")
from src.data_loader import load_golden_set
from src.pipeline import AppleSupportAgent

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

dataset = load_golden_set()
agent = AppleSupportAgent(use_llm=False)

misclassified = []
esc_errors = []

for d in dataset:
    resp = agent.process_tweet(d["customer_text"])
    if resp.intent != d["ground_truth_intent"]:
        misclassified.append((d["customer_text"], d["ground_truth_intent"], resp.intent, d["sampling_stratum"]))
    if resp.should_escalate != d["ground_truth_escalate"]:
        esc_errors.append((d["customer_text"], d["ground_truth_escalate"], resp.should_escalate, resp.escalation_reason))

print(f"Total misclassified intents: {len(misclassified)} / {len(dataset)}")
print(f"Total escalation errors: {len(esc_errors)} / {len(dataset)}")

print("\n--- Sample Intent Misclassifications ---")
for q, true_i, pred_i, stratum in misclassified[:12]:
    print(f"[{stratum}] True: {true_i} | Pred: {pred_i} | \"{q}\"")

print("\n--- Sample Escalation Errors ---")
for q, true_e, pred_e, reason in esc_errors[:8]:
    print(f"True: {true_e} | Pred: {pred_e} | Reason: {reason} | \"{q}\"")
