
import os
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_ai(prompt):
    if not API_KEY:
        raise ValueError(
            "API key missing. Add OPENROUTER_API_KEY to your .env file."
        )

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Workflow Opportunity Finder",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a business workflow analyst. "
                        "Give practical, explainable recommendations. "
                        "Never claim estimates are guaranteed."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1200,
        },
        timeout=90,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: "
            f"{response.text[:700]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]


def analyze_tasks(task_text):
    if not task_text.strip():
        return "Please enter at least one task."

    tasks = []

    for line_number, line in enumerate(task_text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in line.split("|")]

        if len(parts) != 4:
            return (
                f"Line {line_number} has an incorrect format.\n\n"
                "Use: Task name | times per week | minutes each | "
                "error risk (1-5)"
            )

        try:
            name = parts[0]
            frequency = int(parts[1])
            minutes = float(parts[2])
            risk = int(parts[3])

            if not name or frequency < 0 or minutes < 0:
                return f"Line {line_number}: values cannot be negative."
            if not 1 <= risk <= 5:
                return f"Line {line_number}: risk must be between 1 and 5."

            weekly_hours = frequency * minutes / 60
            annual_hours = weekly_hours * 52

            # Transparent heuristic, not a trained ML prediction.
            score = annual_hours * (1 + risk / 5)

            tasks.append({
                "name": name,
                "frequency": frequency,
                "minutes": minutes,
                "risk": risk,
                "weekly_hours": weekly_hours,
                "annual_hours": annual_hours,
                "score": score,
            })

        except ValueError:
            return (
                f"Line {line_number}: frequency, minutes, and risk "
                "must be numbers."
            )

    if not tasks:
        return "Please enter at least one valid task."

    tasks.sort(key=lambda item: item["score"], reverse=True)

    report = [
        "# Workflow Automation Analysis",
        "",
        "| Rank | Task | Hours/week | Hours/year | Risk | Score |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for rank, task in enumerate(tasks, 1):
        report.append(
            f"| {rank} | {task['name']} | "
            f"{task['weekly_hours']:.2f} | "
            f"{task['annual_hours']:.1f} | "
            f"{task['risk']}/5 | {task['score']:.1f} |"
        )

    total_weekly = sum(t["weekly_hours"] for t in tasks)
    total_annual = sum(t["annual_hours"] for t in tasks)

    report.extend([
        "",
        f"**Total time:** {total_weekly:.2f} hours per week "
        f"({total_annual:.1f} hours per year).",
        "",
        "**How ranking works:** annual hours × (1 + risk/5). "
        "A higher score indicates a task worth investigating first. "
        "It is a heuristic, not a trained ML model.",
        "",
        "## AI Recommendations",
    ])

    summary = "\n".join(
        f"- {t['name']}: {t['annual_hours']:.1f} hours/year, "
        f"risk {t['risk']}/5"
        for t in tasks
    )

    prompt = f"""
Analyze these user-provided workflow tasks:

{summary}

For the highest-priority tasks, explain:
1. Why each task may be suitable for automation.
2. A realistic automation idea.
3. Implementation difficulty: Low, Medium, or High.
4. Risks, exceptions, and human checks needed.
5. Suggested first step for validating the idea.

Do not invent company facts. Distinguish estimates from facts.
Return clear headings and a concise recommendation.
"""

    try:
        report.append(call_ai(prompt))
    except Exception as exc:
        report.append(
            "**AI service unavailable.** The calculated ranking is still "
            f"available.\n\nDetails: {str(exc)}"
        )

    return "\n\n".join(report)


with gr.Blocks(title="AI Workflow Opportunity Finder") as demo:
    gr.Markdown(
        "# AI Workflow Automation Opportunity Finder\n"
        "### Find repetitive work worth investigating for automation."
    )

    gr.Markdown(
        "Enter one task per line using this format:\n"
        "`Task name | times per week | minutes each | risk (1-5)`"
    )

    task_input = gr.Textbox(
        label="Workflow task details",
        lines=8,
        placeholder=(
            "Preparing weekly report | 5 | 30 | 3\n"
            "Copying data between sheets | 10 | 12 | 4\n"
            "Checking invoices | 20 | 5 | 2"
        ),
    )

    analyze_button = gr.Button("Analyze Opportunities", variant="primary")
    output = gr.Markdown(label="Analysis")

    analyze_button.click(
        fn=analyze_tasks,
        inputs=task_input,
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
