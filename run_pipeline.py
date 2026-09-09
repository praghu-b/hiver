"""
Interactive CLI Runner for AppleSupport AI Agent.
Demonstrates end-to-end processing: Intent Classification, Grounded Reply Drafting,
and Auto-Handle vs Escalation Triaging.
"""

import sys
from src.pipeline import AppleSupportAgent

# Set utf-8 output encoding for clean terminal display
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_QUERIES = [
    "My iPhone 8 battery drops 30% in an hour after installing iOS 11. What should I do?",
    "Dropped my phone on concrete and the back glass is completely shattered. How much to fix?",
    "I see three unauthorized charges of $29.99 from ITUNES.COM/BILL on my bank account!",
    "Will the Apple Pencil 2 work with my 9.7-inch iPad 2018?",
    "The new iOS 11 Control Center is the worst design Apple has ever released. Fix it!",
    "My iPhone battery is physically expanding and the screen is popping off the frame!"
]

def format_banner():
    print("=" * 75)
    print("        AppleSupport AI Customer Support Agent (Interactive CLI)        ")
    print("=" * 75)

def run_demo():
    format_banner()
    print("Initializing pipeline...")
    agent = AppleSupportAgent(use_llm=True)
    print("Agent ready!\n")

    print("--- Running Sample Queries Across All 5 Intents ---\n")
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"[{i}] Customer Tweet: \"{query}\"")
        resp = agent.process_tweet(query)
        print(f"    Intent:        {resp.intent} (Confidence: {resp.intent_confidence:.2f})")
        esc_str = "ESCALATE TO HUMAN" if resp.should_escalate else "AUTO-HANDLE"
        print(f"    Triage:        {esc_str} (Confidence: {resp.escalation_confidence:.2f})")
        print(f"    Reason:        {resp.escalation_reason}")
        print(f"    Grounded KB:   {resp.grounded_topic} ({resp.grounded_kb_id})")
        print(f"    Draft Reply:   \"{resp.draft_reply}\"")
        print(f"    Length:        {resp.character_count}/280 chars | Latency: {resp.processing_time_ms} ms")
        print("-" * 75)

    print("\nEnter your own customer query (or press Enter to exit):")
    try:
        while True:
            user_input = input("\nCustomer Tweet > ").strip()
            if not user_input:
                break
            resp = agent.process_tweet(user_input)
            print("\n[AI Agent Response]")
            print(f"  * Detected Intent:     {resp.intent} (Confidence: {resp.intent_confidence:.2f})")
            esc_str = "ESCALATE TO HUMAN" if resp.should_escalate else "AUTO-HANDLE"
            print(f"  * Handling Decision:   {esc_str}")
            print(f"  * Stated Reason:       {resp.escalation_reason}")
            print(f"  * Draft Reply:         \"{resp.draft_reply}\"")
            print(f"  * Twitter Length:      {resp.character_count}/280 characters")
            print(f"  * Latency:             {resp.processing_time_ms} ms")
    except (EOFError, KeyboardInterrupt):
        pass
    print("\nExiting. Thank you!")

if __name__ == "__main__":
    run_demo()
