import json
import os
import re
from datetime import datetime
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from markupsafe import escape
from werkzeug.utils import secure_filename

from agent import run_agent
from data_utils import get_dataset_preview, is_allowed_file, read_dataset

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "local-dev-secret-change-before-deploy")

UPLOAD_FOLDER = "uploads"
CHARTS_FOLDER = os.path.join("static", "charts")
REPORTS_FOLDER = os.path.join("static", "reports")
EXAMPLES_FOLDER = "examples"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(EXAMPLES_FOLDER, exist_ok=True)

DEMO_DATASETS = {
    "portfolio": {
        "label": "Творческое портфолио",
        "filename": "creative_portfolio_demo.csv",
        "description": "36 проектов: категории, стили, инструменты, просмотры, лайки, комментарии и сохранения.",
        "instruction": (
            "Проанализируй творческое портфолио: найди сильные и слабые проекты, "
            "недооценённые работы, успешные категории/стили и дай рекомендации по улучшению портфолио."
        ),
    },
    "content": {
        "label": "Контент и соцсети",
        "filename": "social_content_demo.csv",
        "description": "60 публикаций: платформы, форматы, охваты, реакции, клики, расходы и конверсии.",
        "instruction": (
            "Проанализируй эффективность контента: какие платформы, форматы и темы дают лучшие охваты, "
            "вовлечённость, клики и конверсии. Найди слабые публикации и дай рекомендации по контент-стратегии."
        ),
    },
}

DEFAULT_INSTRUCTION = (
    "Проанализируй датасет: найди ключевые закономерности, сильные и слабые места, "
    "аномалии, полезные метрики и дай практические рекомендации."
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "30"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def is_safe_dataset_path(path: str) -> bool:
    if not path:
        return False

    abs_path = os.path.abspath(path)

    allowed_roots = [
        os.path.abspath(UPLOAD_FOLDER),
        os.path.abspath(EXAMPLES_FOLDER),
    ]

    return any(abs_path.startswith(root) for root in allowed_roots)


def get_demo_dataset_info(demo_key: str) -> dict:
    return DEMO_DATASETS.get(demo_key, DEMO_DATASETS["portfolio"])


def validate_dataset(df):
    errors = []
    warnings = []

    if df.empty:
        errors.append("Датасет пустой: в нём нет строк для анализа.")

    if df.shape[1] == 0:
        errors.append("В датасете не найдено колонок.")

    if df.shape[0] < 5:
        warnings.append(
            "В датасете очень мало строк. Анализ будет выполнен, но выводы могут быть менее надёжными."
        )

    if df.shape[1] < 2:
        errors.append("В датасете должно быть минимум 2 колонки, чтобы сравнивать признаки.")

    if df.shape[0] > 50000:
        warnings.append(
            "Файл большой. Анализ будет выполнен по полному датасету, но может занять больше времени."
        )

    numeric_columns = list(df.select_dtypes(include="number").columns)
    text_columns = list(df.select_dtypes(include="object").columns)

    if not numeric_columns:
        warnings.append(
            "В датасете не найдено числовых колонок. Можно будет проанализировать структуру и текстовые поля, "
            "но метрики и графики могут быть ограничены."
        )

    if not text_columns:
        warnings.append(
            "В датасете нет текстовых или категориальных колонок. Анализ будет основан в основном на числовых показателях."
        )

    duplicated_rows = int(df.duplicated().sum())
    if duplicated_rows > 0:
        warnings.append(
            f"Найдены дубликаты строк: {duplicated_rows}. Агент учтёт это при анализе качества данных."
        )

    return errors, warnings


def normalize_report(raw_report: str) -> str:
    if raw_report is None:
        return ""

    text = str(raw_report).strip()

    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)

        if isinstance(data, dict):
            if isinstance(data.get("answer"), str):
                return data["answer"].strip()

            if isinstance(data.get("report"), str):
                return data["report"].strip()

        if isinstance(data, str):
            return normalize_report(data)

    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(0))

            if isinstance(data, dict) and isinstance(data.get("answer"), str):
                return data["answer"].strip()

        except Exception:
            pass

    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    text = text.replace('\\"', '"')

    text = re.sub(r'^\s*\{\s*"tool"\s*:\s*"final"\s*,\s*"answer"\s*:\s*"', "", text)
    text = re.sub(r'"\s*\}\s*$', "", text)

    return text.strip()


def inline_markdown(text: str) -> str:
    safe = str(escape(text))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"`(.+?)`", r"<code>\1</code>", safe)
    return safe


def markdown_table_to_html(table_lines: list[str]) -> str:
    rows = []

    for line in table_lines:
        stripped = line.strip()

        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)

    if len(rows) < 2:
        return "\n".join(f"<p>{inline_markdown(line)}</p>" for line in table_lines)

    headers = rows[0]
    body_rows = rows[2:] if len(rows) > 2 else []

    html = ["<div class='report-table-wrap'><table class='report-table'>"]

    html.append("<thead><tr>")
    for header in headers:
        html.append(f"<th>{inline_markdown(header)}</th>")
    html.append("</tr></thead>")

    html.append("<tbody>")
    for row in body_rows:
        html.append("<tr>")

        for cell in row:
            html.append(f"<td>{inline_markdown(cell)}</td>")

        html.append("</tr>")
    html.append("</tbody>")

    html.append("</table></div>")

    return "\n".join(html)


def report_to_html(text: str) -> str:
    text = normalize_report(text)
    lines = text.splitlines()

    html = []
    in_list = False
    in_pre = False
    pre_lines = []
    table_buffer = []

    def close_list():
        nonlocal in_list

        if in_list:
            html.append("</ul>")
            in_list = False

    def close_pre():
        nonlocal in_pre, pre_lines

        if in_pre:
            html.append("<pre>" + escape("\n".join(pre_lines)) + "</pre>")
            pre_lines = []
            in_pre = False

    def close_table():
        nonlocal table_buffer

        if table_buffer:
            html.append(markdown_table_to_html(table_buffer))
            table_buffer = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            close_table()

            if in_pre:
                close_pre()
            else:
                close_list()
                in_pre = True
                pre_lines = []

            continue

        if in_pre:
            pre_lines.append(line)
            continue

        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line:
            close_list()
            table_buffer.append(stripped)
            continue
        else:
            close_table()

        if not stripped:
            close_list()
            html.append("<div class='report-gap'></div>")
            continue

        if stripped in {"---", "***", "___"}:
            close_list()
            html.append("<hr class='report-divider'>")
            continue

        if stripped.startswith(">"):
            close_list()
            quote_text = stripped.lstrip(">").strip()
            html.append(f"<blockquote>{inline_markdown(quote_text)}</blockquote>")
            continue

        if stripped.startswith("### "):
            close_list()
            html.append(f"<h3>{inline_markdown(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            close_list()
            html.append(f"<h2>{inline_markdown(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            close_list()
            html.append(f"<h2>{inline_markdown(stripped[2:])}</h2>")
        elif re.match(r"^[-*•]\s+", stripped):
            if not in_list:
                html.append("<ul>")
                in_list = True

            item = re.sub(r"^[-*•]\s+", "", stripped)
            html.append(f"<li>{inline_markdown(item)}</li>")
        elif re.match(r"^\d+\.\s+", stripped):
            close_list()
            html.append(f"<p>{inline_markdown(stripped)}</p>")
        else:
            close_list()
            html.append(f"<p>{inline_markdown(stripped)}</p>")

    close_table()
    close_list()
    close_pre()

    return "\n".join(html)


def build_result_cards(report: str) -> list[dict[str, str]]:
    text = normalize_report(report)
    clean = re.sub(r"\s+", " ", text).strip()

    def find_text(markers: list[str], fallback: str) -> str:
        lower = clean.lower()

        for marker in markers:
            index = lower.find(marker.lower())

            if index != -1:
                fragment = clean[index:index + 750]
                fragment = re.sub(r"\|.*?\|", " ", fragment)
                fragment = re.sub(r"[#*_`>|]", " ", fragment)
                fragment = re.sub(r"\s+", " ", fragment).strip()

                sentences = re.split(r"(?<=[.!?])\s+", fragment)

                for sentence in sentences:
                    sentence = sentence.strip(" -—:;")

                    if 70 <= len(sentence) <= 300:
                        return sentence

                if len(fragment) > 80:
                    return fragment[:270].rsplit(" ", 1)[0] + "..."

        return fallback

    return [
        {
            "title": "Главный вывод",
            "text": find_text(
                ["Вывод:", "итоговый отчёт", "анализ датасета", "анализ творческого портфолио"],
                "Данные проанализированы: найдены ключевые закономерности, сильные и слабые места, а также практические рекомендации.",
            ),
        },
        {
            "title": "Что работает лучше",
            "text": find_text(
                ["сильные проекты", "успешные категории", "топ-3", "лучшие", "лидеры"],
                "В отчёте выделены признаки, категории или элементы, которые показывают лучшие результаты.",
            ),
        },
        {
            "title": "Что стоит улучшить",
            "text": find_text(
                ["слабые проекты", "слабые", "недооценённые", "аномалии", "проблема"],
                "Отдельно отмечены слабые места, возможные аномалии и элементы, которым требуется доработка.",
            ),
        },
        {
            "title": "Что сделать дальше",
            "text": find_text(
                ["рекомендации", "следующий шаг", "улучшить", "продвигать"],
                "Используйте рекомендации из полного отчёта, чтобы принять решения по улучшению результата.",
            ),
        },
    ]


def create_fallback_chart(df, charts_dir: str) -> None:
    os.makedirs(charts_dir, exist_ok=True)

    numeric_columns = list(df.select_dtypes(include="number").columns)

    if not numeric_columns:
        return

    chart_path = os.path.join(charts_dir, "fallback_metrics.png")

    plt.rcParams.update(
        {
            "figure.facecolor": "#0B1020",
            "axes.facecolor": "#111827",
            "axes.edgecolor": "#263244",
            "axes.labelcolor": "#E5E7EB",
            "xtick.color": "#CBD5E1",
            "ytick.color": "#CBD5E1",
            "text.color": "#F8FAFC",
            "axes.titlecolor": "#F8FAFC",
            "grid.color": "#FFFFFF",
            "grid.alpha": 0.08,
            "font.size": 11,
        }
    )

    plot_df = df.copy()

    name_column = None
    for possible in ["project_name", "projectname", "post_id", "name", "title", "project", "название"]:
        if possible in plot_df.columns:
            name_column = possible
            break

    if "views" in plot_df.columns and name_column:
        top = plot_df.sort_values("views", ascending=False).head(10)

        plt.figure(figsize=(11, 6))
        plt.barh(top[name_column].astype(str), top["views"])
        plt.gca().invert_yaxis()
        plt.title("Топ записей по просмотрам")
        plt.xlabel("Просмотры")
        plt.grid(axis="x", alpha=0.12)

    elif "impressions" in plot_df.columns and name_column:
        top = plot_df.sort_values("impressions", ascending=False).head(10)

        plt.figure(figsize=(11, 6))
        plt.barh(top[name_column].astype(str), top["impressions"])
        plt.gca().invert_yaxis()
        plt.title("Топ записей по показам")
        plt.xlabel("Показы")
        plt.grid(axis="x", alpha=0.12)

    else:
        means = plot_df[numeric_columns].mean().sort_values(ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(means.index.astype(str), means.values)
        plt.title("Средние значения числовых показателей")
        plt.xlabel("Среднее значение")
        plt.grid(axis="x", alpha=0.12)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=170, bbox_inches="tight")
    plt.close()


def save_report_files(session_id: str, payload: dict) -> dict:
    txt_path = os.path.join(REPORTS_FOLDER, f"{session_id}_report.txt")
    json_path = os.path.join(REPORTS_FOLDER, f"{session_id}_report.json")

    report = normalize_report(payload.get("report", ""))

    text = (
        "Creative Portfolio Analyst\n"
        f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Файл: {payload['dataset_name']}\n"
        f"Размер: {payload['shape'][0]} строк, {payload['shape'][1]} колонок\n\n"
        "Что нужно было проанализировать:\n"
        f"{payload['instruction']}\n\n"
        "Итоговый отчёт:\n"
        f"{report}\n\n"
        "----------------------------------------\n"
        "Журнал действий агента\n"
        "----------------------------------------\n"
        "Этот раздел нужен для проверки задания и показывает, какие шаги выполнялись во время анализа.\n"
    )

    for item in payload["action_log"]:
        text += f"\nШаг {item.get('step')} — {item.get('tool')}\n"

        if item.get("reason"):
            text += f"Причина: {item.get('reason')}\n"

        if item.get("code"):
            text += f"Код:\n{item.get('code')}\n"

        if item.get("result"):
            text += f"Результат:\n{item.get('result')}\n"

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(text)

    clean_payload = dict(payload)
    clean_payload["report"] = report

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(clean_payload, file, ensure_ascii=False, indent=2)

    return {
        "txt": txt_path,
        "json": json_path,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        default_instruction=DEFAULT_INSTRUCTION,
        demo_datasets=DEMO_DATASETS,
        max_upload_mb=MAX_UPLOAD_MB,
        model=os.getenv("OPENROUTER_MODEL", "not specified"),
    )


@app.route("/preview", methods=["POST"])
def preview():
    instruction = request.form.get("instruction", "").strip()
    use_demo = request.form.get("use_demo") == "on"
    demo_key = request.form.get("demo_dataset", "portfolio")
    file = request.files.get("dataset")

    session_id = str(uuid4())

    try:
        if use_demo:
            demo_info = get_demo_dataset_info(demo_key)
            dataset_path = os.path.join(EXAMPLES_FOLDER, demo_info["filename"])
            dataset_name = demo_info["filename"]

            if not os.path.exists(dataset_path):
                flash(
                    "Демо-датасет не найден. Запустите команду: python make_demo_datasets.py"
                )
                return redirect(url_for("index"))

            if not instruction:
                instruction = demo_info["instruction"]

        else:
            instruction = instruction or DEFAULT_INSTRUCTION

            if not file or file.filename == "":
                flash("Загрузите CSV/XLSX файл или выберите демо-датасет.")
                return redirect(url_for("index"))

            if not is_allowed_file(file.filename):
                flash("Неверный формат файла. Поддерживаются только CSV и XLSX.")
                return redirect(url_for("index"))

            safe_name = secure_filename(file.filename)

            if not safe_name:
                flash("Некорректное имя файла.")
                return redirect(url_for("index"))

            dataset_name = safe_name
            dataset_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{safe_name}")
            file.save(dataset_path)

        df = read_dataset(dataset_path)
        errors, warnings = validate_dataset(df)

        if errors:
            for error in errors:
                flash(error)

            return redirect(url_for("index"))

        preview_data = get_dataset_preview(df, rows=12)

        return render_template(
            "preview.html",
            session_id=session_id,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            instruction=instruction,
            shape=preview_data["shape"],
            columns=preview_data["columns"],
            preview_html=preview_data["preview_html"],
            warnings=warnings,
            model=os.getenv("OPENROUTER_MODEL", "not specified"),
        )

    except Exception as e:
        flash(f"Не удалось прочитать файл: {e}")
        return redirect(url_for("index"))


@app.route("/analyze", methods=["POST"])
def analyze():
    session_id = request.form.get("session_id", "").strip()
    dataset_path = request.form.get("dataset_path", "").strip()
    dataset_name = request.form.get("dataset_name", "").strip() or "dataset"
    instruction = request.form.get("instruction", "").strip() or DEFAULT_INSTRUCTION

    if not session_id:
        session_id = str(uuid4())

    if not is_safe_dataset_path(dataset_path):
        flash("Путь к датасету некорректен. Загрузите файл заново.")
        return redirect(url_for("index"))

    try:
        df = read_dataset(dataset_path)
        errors, _warnings = validate_dataset(df)

        if errors:
            for error in errors:
                flash(error)

            return redirect(url_for("index"))

        preview_data = get_dataset_preview(df, rows=10)

        charts_dir = os.path.join(CHARTS_FOLDER, session_id)
        os.makedirs(charts_dir, exist_ok=True)

        result = run_agent(df, instruction, charts_dir)
        action_log = result["action_log"]

        clean_report = normalize_report(result["report"])

        chart_files = []
        if os.path.exists(charts_dir):
            for name in sorted(os.listdir(charts_dir)):
                if name.lower().endswith((".png", ".jpg", ".jpeg")):
                    chart_files.append(f"charts/{session_id}/{name}")

        if not chart_files:
            create_fallback_chart(df, charts_dir)

            if os.path.exists(charts_dir):
                for name in sorted(os.listdir(charts_dir)):
                    if name.lower().endswith((".png", ".jpg", ".jpeg")):
                        chart_files.append(f"charts/{session_id}/{name}")

        python_steps = sum(1 for item in action_log if item.get("tool") == "python")
        parse_errors = sum(1 for item in action_log if item.get("tool") == "parse_error")

        payload = {
            "session_id": session_id,
            "dataset_name": dataset_name,
            "instruction": instruction,
            "shape": preview_data["shape"],
            "columns": preview_data["columns"],
            "report": clean_report,
            "action_log": action_log,
            "chart_files": chart_files,
            "python_steps": python_steps,
            "parse_errors": parse_errors,
            "model": os.getenv("OPENROUTER_MODEL", "not specified"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        save_report_files(session_id, payload)

        return render_template(
            "report.html",
            session_id=session_id,
            dataset_name=dataset_name,
            instruction=instruction,
            shape=preview_data["shape"],
            columns=preview_data["columns"],
            preview_html=preview_data["preview_html"],
            report=clean_report,
            report_html=report_to_html(clean_report),
            result_cards=build_result_cards(clean_report),
            action_log=action_log,
            chart_files=chart_files,
            python_steps=python_steps,
            parse_errors=parse_errors,
            model=os.getenv("OPENROUTER_MODEL", "not specified"),
        )

    except Exception as e:
        flash(f"Ошибка анализа: {e}")
        return redirect(url_for("index"))


@app.route("/download/<session_id>/<file_type>")
def download_report(session_id: str, file_type: str):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", session_id):
        flash("Некорректный идентификатор отчёта.")
        return redirect(url_for("index"))

    if file_type not in {"txt", "json"}:
        flash("Неподдерживаемый тип файла.")
        return redirect(url_for("index"))

    path = os.path.join(REPORTS_FOLDER, f"{session_id}_report.{file_type}")

    if not os.path.exists(path):
        flash("Файл отчёта не найден.")
        return redirect(url_for("index"))

    return send_file(path, as_attachment=True)


@app.errorhandler(413)
def too_large(_error):
    flash(f"Файл слишком большой. Максимальный размер: {MAX_UPLOAD_MB} МБ.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)