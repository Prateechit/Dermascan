"""
AI health chatbot.

Two responsibilities:
  1. Ask the user 4-5 short health-screening questions (itching, pain,
     duration, spreading, age).
  2. Combine those answers with the model's prediction to produce plain-language
     guidance.

The guidance engine is rule-based by default so the project runs with zero API
keys and works offline. If an Anthropic API key is present in the environment
(ANTHROPIC_API_KEY), `generate_ai_reply()` can instead ask Claude for a more
natural, generative response \u2014 satisfying the "Generative AI" requirement while
keeping a safe, deterministic fallback.
"""

import os

# The fixed screening questionnaire shown by the UI.
QUESTIONS = [
    {"id": "itching", "text": "Is the affected area itchy?",
     "options": ["No", "Mild", "Severe"]},
    {"id": "pain", "text": "Do you feel any pain or tenderness there?",
     "options": ["No", "Sometimes", "Yes, constantly"]},
    {"id": "duration", "text": "How long have you noticed this?",
     "options": ["Less than a week", "1\u20134 weeks", "More than a month"]},
    {"id": "spreading", "text": "Is it changing in size, colour, or spreading?",
     "options": ["No", "Not sure", "Yes"]},
    {"id": "age", "text": "What is your age group?",
     "options": ["Under 18", "18\u201340", "41\u201360", "Over 60"]},
]


def _risk_from_answers(answers):
    """Crude symptom-risk score from questionnaire answers (0-6)."""
    score = 0
    if answers.get("itching") == "Severe":
        score += 1
    if answers.get("pain") in ("Sometimes", "Yes, constantly"):
        score += 1
    if answers.get("duration") == "More than a month":
        score += 1
    if answers.get("spreading") == "Yes":
        score += 2
    if answers.get("age") in ("41\u201360", "Over 60"):
        score += 1
    return score


def generate_reply(prediction, answers):
    """Rule-based combined guidance from prediction + answers."""
    symptom_score = _risk_from_answers(answers)
    malignant = prediction.get("malignant", False)
    severity = prediction.get("severity", "low")

    lines = []
    disease = prediction.get("disease_name", "the detected condition")
    conf = prediction.get("confidence", 0)
    lines.append(
        f"Based on the image, the model suggests **{disease}** "
        f"(about {conf}% confidence)."
    )

    if severity in ("critical", "high") or malignant:
        urgency = "high"
        lines.append(
            "This type of lesion can be serious, so you should consult a "
            "dermatologist as soon as possible for a proper examination and, "
            "if needed, a biopsy."
        )
    elif symptom_score >= 3:
        urgency = "medium"
        lines.append(
            "Your answers suggest a few concerning signs. While this condition "
            "is usually not dangerous, it would be wise to have a dermatologist "
            "check it within the next week or two."
        )
    else:
        urgency = "low"
        lines.append(
            "This appears to be a low-risk condition and your symptoms are "
            "mild. Keep monitoring it and see a doctor if anything changes."
        )

    if answers.get("spreading") == "Yes":
        lines.append(
            "Because you mentioned it is changing or spreading, please don't "
            "delay getting it looked at \u2014 changes over time are an important "
            "warning sign."
        )

    lines.append(
        "Meanwhile, avoid scratching the area, protect it from the sun, and "
        "use sunscreen regularly."
    )
    lines.append(
        "\u26a0\ufe0f This is an educational screening tool, not a medical diagnosis."
    )

    return {"urgency": urgency, "message": "\n\n".join(lines)}


def generate_ai_reply(prediction, answers, user_question):
    """Optional: use Claude for a free-form generative answer.

    Falls back to the rule-based reply if no API key is configured or the
    request fails. This keeps the app fully functional offline.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        base = generate_reply(prediction, answers)
        return base["message"]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        context = (
            f"Predicted condition: {prediction.get('disease_name')} "
            f"({prediction.get('confidence')}% confidence, "
            f"severity {prediction.get('severity')}).\n"
            f"Questionnaire answers: {answers}.\n"
            f"User question: {user_question or 'What should I do now?'}"
        )
        system = (
            "You are a cautious dermatology assistant. Give brief, plain-language "
            "guidance. Never give a definitive diagnosis. Always recommend seeing "
            "a real dermatologist for anything concerning. Keep it under 120 words."
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": context}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
    except Exception:  # noqa: BLE001
        return generate_reply(prediction, answers)["message"]
