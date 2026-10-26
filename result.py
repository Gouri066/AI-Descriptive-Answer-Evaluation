
import re


def identify_topic(row):
    question = str(row.get("question") or "").lower()
    question_type = str(row.get("question_type") or "").lower()

    # Time and calendar concepts
    if "hour" in question or "minute" in question:
        return "Time and Units"

    if "day" in question or "week" in question or "month" in question or "year" in question:
        return "Time and Calendar"

    # Basic arithmetic, when an expression is explicitly present
    if re.search(r"\d+\s*\+\s*\d+", question) or "addition" in question:
        return "Addition"

    if re.search(r"\d+\s*[-−]\s*\d+", question) or "subtraction" in question:
        return "Subtraction"

    if re.search(r"\d+\s*[×*x]\s*\d+", question) or "multiplication" in question:
        return "Multiplication"

    if re.search(r"\d+\s*[÷/]\s*\d+", question) or "division" in question:
        return "Division"

    if "fill" in question_type and "blank" in question_type:
        return "Fill in the Blank"

    return "General"




def make_practice(topic, original_question, correct_answer):
    if topic == "Time and Units":
        q = (original_question or "").lower()

        if "hour" in q and "day" in q:
            return "How many hours are there in one day?"

        if "minute" in q and "hour" in q:
            return "How many minutes are there in one hour?"

        return "Convert one hour into minutes and explain your answer."

    if topic == "Time and Calendar":
        q = (original_question or "").lower()

        if "day" in q and "week" in q:
            return "How many days are there in one week?"

        if "month" in q and "year" in q:
            return "How many months are there in one year?"

        if "hour" in q and "day" in q:
            return "How many hours are there in one day?"

        return "Write three facts about days, weeks, months and years."

    if topic == "Addition":
        return "Solve: 27 + 16 = _____"

    if topic == "Subtraction":
        return "Solve: 52 - 28 = _____"

    if topic == "Multiplication":
        return "Solve: 7 × 8 = _____"

    if topic == "Division":
        return "Solve: 56 ÷ 7 = _____"

    if original_question:
        return (
            "Try this question again and explain your reasoning: "
            + original_question
        )

    return "Review the original question with your teacher."


def build_phase9(qa_rows, student_name="Student"):
    topic_stats = {}
    weak_questions = []
    strengths = []

    for row in qa_rows:
        topic = identify_topic(row)
        awarded = float(row.get("marks_awarded") or 0)
        maximum = float(row.get("max_marks") or 0)
        correct = bool(row.get("is_correct"))
        question = str(row.get("question") or "Question")

        stats = topic_stats.setdefault(
            topic, {"awarded": 0.0, "maximum": 0.0, "count": 0}
        )
        stats["awarded"] += awarded
        stats["maximum"] += maximum
        stats["count"] += 1

        if not correct or (maximum > 0 and awarded < maximum):
            weak_questions.append({
                "question": question,
                "topic": topic,
                "question_no": row.get("question_no"),
                "correct_answer": row.get("correct_answer", ""),
                "marks_awarded": awarded,
                "max_marks": maximum,
                "feedback": row.get("feedback", ""),
            })
        else:
            strengths.append(topic)

    topic_performance = []

    for topic, stats in topic_stats.items():
        maximum = stats["maximum"]
        percentage = (
            round(stats["awarded"] / maximum * 100)
            if maximum > 0 else 0
        )

        topic_performance.append({
            "topic": topic,
            "percentage": percentage,
            "awarded": round(stats["awarded"], 2),
            "maximum": round(maximum, 2),
            "count": stats["count"],
        })

    topic_performance.sort(key=lambda item: item["percentage"])

    recommendations = []

    for item in topic_performance:
        if item["percentage"] < 60:
            priority = "High"
            action = (
                f"Focus on {item['topic']}. Re-teach the basic concept, "
                "demonstrate worked examples, ask the student to explain "
                "each step, and provide guided practice."
            )
        elif item["percentage"] < 80:
            priority = "Medium"
            action = (
                f"Revise {item['topic']}. Work through examples together, "
                "identify mistakes, and assign additional practice."
            )
        else:
            continue

        recommendations.append({
            "topic": item["topic"],
            "priority": priority,
            "action": action,
        })

    if not recommendations:
        recommendations.append({
            "topic": "Continue learning",
            "priority": "Low",
            "action": (
                "The student performed well on the assessed questions. "
                "Continue with revision and gradually introduce harder questions."
            ),
        })

    worksheet = []

    for index, item in enumerate(weak_questions, start=1):
        worksheet.append({
            "number": index,
            "topic": item["topic"],
            "question": make_practice(
                item["topic"],
                item["question"],
                item["correct_answer"],
            ),
            "original_question": item["question"],
            "correct_answer": item["correct_answer"],
        })

    total_awarded = sum(
        float(row.get("marks_awarded") or 0) for row in qa_rows
    )
    total_maximum = sum(
        float(row.get("max_marks") or 0) for row in qa_rows
    )

    percentage = (
        round(total_awarded / total_maximum * 100)
        if total_maximum > 0 else 0
    )

    weak_topics = sorted({
        item["topic"] for item in weak_questions
    })

    if percentage >= 80:
        feedback = (
            f"Well done, {student_name}! You have shown good performance. "
            "Keep practising and challenge yourself with more advanced questions."
        )
    elif percentage >= 60:
        feedback = (
            f"Good effort, {student_name}! Review "
            + (", ".join(weak_topics) if weak_topics else "the assessed concepts")
            + ". Practise regularly and check your answers carefully."
        )
    elif percentage >= 40:
        feedback = (
            f"Keep going, {student_name}! Focus on "
            + (", ".join(weak_topics) if weak_topics else "the basic concepts")
            + ". Work through examples step by step and ask your teacher "
            "for help where needed."
        )
    else:
        feedback = (
            f"Don't be discouraged, {student_name}. Start with the "
            "fundamentals, practise with guided examples, and review "
            "each mistake with your teacher."
        )

    return {
        "topic_performance": topic_performance,
        "recommendations": recommendations,
        "weak_questions": weak_questions,
        "strengths": sorted(set(strengths)),
        "worksheet": worksheet,
        "feedback": feedback,
        "percentage": percentage,
        "total_awarded": round(total_awarded, 2),
        "total_maximum": round(total_maximum, 2),
    }
