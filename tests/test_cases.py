"""
tests/test_cases.py - the complaints we test the system with.

Cases 1-6 come from the assignment PDF. Cases 7-9 are our own extra cases
(the assignment asks for at least two of your own).
"""

TEST_CASES = [
    {
        "id": 1,
        "complaint": "Where is my order? It was supposed to arrive yesterday.",
        "customer_name": "Priya",
        "order_id": "ORD-4512",
        "expected_category": {"Shipping"},
        "expected_status": "Resolved",
    },
    {
        "id": 2,
        "complaint": "I want to return the shoes I bought last week. How do I get a refund?",
        "customer_name": "Arjun",
        "order_id": "ORD-4513",
        "expected_category": {"Refund"},
        "expected_status": "Resolved",
    },
    {
        "id": 3,
        "complaint": "I forgot my password and can't log in.",
        "customer_name": "Meena",
        "order_id": "",
        "expected_category": {"Account"},
        "expected_status": "Resolved",
    },
    {
        "id": 4,
        "complaint": "The blender I received is broken. Can I get a replacement?",
        "customer_name": "Karthik",
        "order_id": "ORD-4520",
        "expected_category": {"Product", "Refund"},
        "expected_status": "Resolved",
    },
    {
        "id": 5,
        "complaint": "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.",
        "customer_name": "Rahul",
        "order_id": "ORD-4533",
        "expected_category": {"Billing"},
        "expected_urgency": "High",
        "expected_sentiment": "Angry",
        "expected_status": "Escalated",
    },
    {
        "id": 6,
        "complaint": "Do you sell gift cards for a company of 500 employees with custom branding?",
        "customer_name": "Anitha",
        "order_id": "",
        "expected_category": {"Other", "Product"},
        "expected_status": "Escalated",
    },
    # ---- Our own test cases ----
    {
        "id": 7,
        "complaint": "Someone hacked my account. There are two orders I never placed and my address was changed.",
        "customer_name": "Vikram",
        "order_id": "",
        "expected_category": {"Account"},
        "expected_urgency": "High",
        "expected_status": "Escalated",
    },
    {
        "id": 8,
        "complaint": "என் ஆர்டர் இன்னும் வரவில்லை, 5 நாட்கள் ஆகிவிட்டது. தயவுசெய்து உதவுங்கள்.",
        "customer_name": "Lakshmi",
        "order_id": "ORD-4550",
        "expected_category": {"Shipping"},
        "expected_language": "Tamil",
        "expected_status": "Resolved",
    },
    {
        "id": 9,
        "complaint": "Hi, can I still change the delivery address? My order has not shipped yet. Thanks!",
        "customer_name": "Sneha",
        "order_id": "ORD-4561",
        "expected_category": {"Shipping"},
        "expected_urgency": "Low",
        "expected_status": "Resolved",
    },
]
