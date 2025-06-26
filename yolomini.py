import gradio as gr
import cv2
import os
import numpy as np
from ultralytics import YOLO
from datetime import date
import csv
import pandas as pd
import os

model = YOLO("yolov8n.pt")

schedule = {
    "1": {
        "110": {"teacher": "Петров А.А.", "capacity": 100, "group": "ИВТ-21", "discipline": "Базы данных"},
        "120": {"teacher": "Сидорова М.В.", "capacity": 90, "group": "ПИ-22", "discipline": "Алгоритмы"},
        "130": {"teacher": "Козлов Д.С.", "capacity": 35, "group": "ФИЗ-23", "discipline": "Квантовая физика"},
        "140": {"teacher": "Иванова Е.П.", "capacity": 15, "group": "ЛИНГ-24", "discipline": "Фонетика"},
    },
    "2": {
        "110": {"teacher": "Смирнов В.Г.", "capacity": 180, "group": "ЭКО-21", "discipline": "Макроэкономика"},
        "120": {"teacher": "Лебедева О.Н.", "capacity": 95, "group": "МАТ-22", "discipline": "Мат. анализ"},
        "130": {"teacher": "Волков П.К.", "capacity": 30, "group": "ХИМ-23", "discipline": "Органическая химия"},
        "140": {"teacher": "Григорьева Т.М.", "capacity": 18, "group": "ПСИ-24", "discipline": "Когнитивная психология"},
    },
    "3": {
        "110": {"teacher": "Петров А.А.", "capacity": 110, "group": "ИВТ-22", "discipline": "Программирование"},
        "120": {"teacher": "Козлов Д.С.", "capacity": 85, "group": "ФИЗ-21", "discipline": "Теор. механика"},
        "130": {"teacher": "Иванова Е.П.", "capacity": 35, "group": "ЛИНГ-23", "discipline": "Синтаксис"},
        "140": {"teacher": "Белова Л.Р.", "capacity": 20, "group": "МЕД-24", "discipline": "Анатомия"},
    },
    "4": {
        "110": {"teacher": "Сидорова М.В.", "capacity": 190, "group": "ПИ-21", "discipline": "Искусственный интеллект"},
        "120": {"teacher": "Смирнов В.Г.", "capacity": 95, "group": "ЭКО-22", "discipline": "Эконометрика"},
        "130": {"teacher": "Григорьева Т.М.", "capacity": 38, "group": "ПСИ-23", "discipline": "Социальная психология"},
        "140": {"teacher": "Волков П.К.", "capacity": 20, "group": "ХИМ-24", "discipline": "Биохимия"},
    },
    "5": {
        "110": {"teacher": "Лебедева О.Н.", "capacity": 170, "group": "МАТ-21", "discipline": "Дифференциальные уравнения"},
        "120": {"teacher": "Петров А.А.", "capacity": 90, "group": "ИВТ-23", "discipline": "Веб-разработка"},
        "130": {"teacher": "Козлов Д.С.", "capacity": 40, "group": "ФИЗ-24", "discipline": "Астрофизика"},
        "140": {"teacher": "Иванова Е.П.", "capacity": 15, "group": "ЛИНГ-21", "discipline": "Фонология"},
    },
    "6": {
        "110": {"teacher": "Сидорова М.В.", "capacity": 110, "group": "ПИ-23", "discipline": "Машинное обучение"},
        "120": {"teacher": "Смирнов В.Г.", "capacity": 95, "group": "ЭКО-23", "discipline": "Финансы"},
        "130": {"teacher": "Белова Л.Р.", "capacity": 35, "group": "МЕД-21", "discipline": "Гистология"},
        "140": {"teacher": "Григорьева Т.М.", "capacity": 18, "group": "ПСИ-22", "discipline": "Психология личности"},
    },
    "7": {
        "110": {"teacher": "Петров А.А.", "capacity": 90, "group": "ИВТ-24", "discipline": "СУБД"},
        "120": {"teacher": "Лебедева О.Н.", "capacity": 80, "group": "МАТ-23", "discipline": "Теория вероятностей"},
        "130": {"teacher": "Волков П.К.", "capacity": 30, "group": "ХИМ-22", "discipline": "Физическая химия"},
        "140": {"teacher": "Иванова Е.П.", "capacity": 15, "group": "ЛИНГ-22", "discipline": "Морфология"},
    },
    "8": {
        "110": {"teacher": "Козлов Д.С.", "capacity": 70, "group": "ФИЗ-22", "discipline": "Оптика"},
        "120": {"teacher": "Григорьева Т.М.", "capacity": 60, "group": "ПСИ-21", "discipline": "Общая психология"},
        "130": {"teacher": "Смирнов В.Г.", "capacity": 30, "group": "ЭКО-24", "discipline": "Экономика"},
        "140": {"teacher": "Белова Л.Р.", "capacity": 15, "group": "МЕД-23", "discipline": "Физиология"},
    },
}

users = {
    "petrov": {"password": "1111", "name": "Петров А.А."},
    "sidorova": {"password": "1111", "name": "Сидорова М.В."},
    "kozlov": {"password": "1111", "name": "Козлов Д.С."},
    "ivanova": {"password": "1111", "name": "Иванова Е.П."},
    "smirnov": {"password": "1111", "name": "Смирнов В.Г."},
    "lebedeva": {"password": "1111", "name": "Лебедева О.Н."},
    "grigorieva": {"password": "1111", "name": "Григорьева Т.М."},
    "belova": {"password": "1111", "name": "Белова Л.Р."},
    "volkov": {"password": "1111", "name": "Волков П.К."},
    "admin": {"password": "admin123", "name": "Администратор"},
}

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    return sharpened

def generate_admin_report(selected_date):
    try:
        d = date.fromisoformat(selected_date)
        date_str = d.strftime("%d.%m.%Y")
    except ValueError:
        return "❌ Неверный формат даты. Используйте YYYY-MM-DD."

    report_lines = [f"Отчет по всем фото за {date_str}:\n"]

    found_any = False
    rows = []  # для таблицы Excel

    for pair_num, auds in schedule.items():
        for auditorium, info in auds.items():
            filename = f"photo{auditorium}_{date_str}({pair_num}).jpg"
            photo_path = os.path.join("photos", filename)

            if os.path.exists(photo_path):
                image = cv2.imread(photo_path)
                if image is not None:
                    enhanced = enhance_image(image)
                    results = model(enhanced)
                    person_count = sum(1 for r in results for box in r.boxes if int(box.cls[0]) == 0)
                else:
                    person_count = "Ошибка загрузки фото"
            else:
                person_count = "Фото не найдено"

            found_any = True
            line = (
                f"Пара {pair_num}, Аудитория {auditorium}, "
                f"Преподаватель: {info['teacher']}, "
                f"Группа: {info['group']}, "
                f"Дисциплина: {info['discipline']}, "
                f"Вместимость: {info['capacity']}, "
                f"Найдено людей: {person_count}"
            )
            report_lines.append(line)

            # Добавляем строку для Excel
            rows.append({
                "Дата": date_str,
                "Пара": pair_num,
                "Аудитория": auditorium,
                "Преподаватель": info['teacher'],
                "Группа": info['group'],
                "Дисциплина": info['discipline'],
                "Вместимость": info['capacity'],
                "Найдено людей": person_count
            })

    if not found_any:
        return "⚠️ Фото для выбранной даты не найдены."

    # Сохраняем в Excel
    df = pd.DataFrame(rows)
    excel_filename = f"report_{selected_date}.xlsx"
    excel_path = os.path.join("reports", excel_filename)

    os.makedirs("reports", exist_ok=True)
    df.to_excel(excel_path, index=False)

    report_lines.append(f"\nОтчет также сохранен в файл: {excel_path}")

    return "\n".join(report_lines)

def process_photo(auditorium, selected_date, pair_number):
    try:
        d = date.fromisoformat(selected_date)
        date_str = d.strftime("%d.%m.%Y")
    except ValueError:
        return None, "❌ Неверный формат даты. Используйте YYYY-MM-DD."

    filename = f"photo{auditorium}_{date_str}({pair_number}).jpg"
    photo_path = os.path.join("photos", filename)

    if not os.path.exists(photo_path):
        return None, f"❌ Фото не найдено: {filename}"

    image = cv2.imread(photo_path)
    if image is None:
        return None, "❌ Не удалось загрузить изображение."

    enhanced = enhance_image(image)
    results = model(enhanced)

    person_count = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(enhanced, "Person", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    capacity = int(schedule.get(pair_number, {}).get(auditorium, {}).get("capacity", 0))
    percent = (person_count / capacity) * 100 if capacity > 0 else 0
    summary = (f"🧍 Найдено людей: {person_count}\n"
               f"💺 Вместимость аудитории: {capacity}\n"
               f"📊 Заполненность: {percent:.1f}%")

    cv2.putText(enhanced, f"People: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), summary

def get_teacher_pairs(teacher_name):
    allowed = []
    for pair_num, auds in schedule.items():
        for aud_num, info in auds.items():
            if info["teacher"] == teacher_name:
                allowed.append((pair_num, aud_num))
    return allowed

def load_pair_data(pair_num, auditorium):
    if pair_num and auditorium and pair_num in schedule and auditorium in schedule[pair_num]:
        info = schedule[pair_num][auditorium]
        return info.get("teacher", ""), info.get("group", ""), info.get("discipline", ""), info.get("capacity", 0)
    else:
        return "", "", "", 0

def save_schedule_change(pair_num, auditorium, teacher, group, discipline, capacity):
    if pair_num and auditorium:
        schedule.setdefault(pair_num, {})[auditorium] = {
            "teacher": teacher,
            "group": group,
            "discipline": discipline,
            "capacity": int(capacity)
        }
        allowed_pairs, choices = get_all_pairs_choices()
        return "✅ Изменения сохранены", allowed_pairs, gr.update(choices=choices, value=choices[0])
    else:
        return "❌ Пара и аудитория должны быть выбраны", gr.State(), gr.update()

def try_login(user, pwd):
    if user in users and users[user]["password"] == pwd:
        teacher_name = users[user]["name"]
        if user == "admin":
            allowed_pairs = []
            for pair_num, auds in schedule.items():
                for aud_num in auds.keys():
                    allowed_pairs.append((pair_num, aud_num))
            choices = [f"Пара {p}, Аудитория {a}" for p, a in allowed_pairs]
            return (
                gr.update(visible=False),  # login_container
                gr.update(visible=True),  # main_container
                "",  # login_status
                teacher_name,  # teacher_name_state
                allowed_pairs,  # для внутреннего состояния (если нужно)
                gr.update(choices=choices, value=choices[0]),  # pairs_dropdown
                gr.update(visible=True)  # admin_edit_container (добавь этот в outputs)
            )
        else:
            allowed_pairs = get_teacher_pairs(teacher_name)
            choices = [f"Пара {p}, Аудитория {a}" for p, a in allowed_pairs]
            if not choices:
                choices = ["Нет доступных пар"]
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                "",
                teacher_name,
                allowed_pairs,
                gr.update(choices=choices, value=choices[0]),
                gr.update(visible=False)  # скрываем админ-блок
            )
    else:
    # Для обычного пользователя не показываем админский контейнер
        return (
            gr.update(visible=True),  # login_container остаётся
            gr.update(visible=False),  # main_container скрыт
            "Неверный логин или пароль",
            "",
            [],  # пустое состояние
            gr.update(choices=[], value=None),
            gr.update(visible=False)  # admin блок скрыт
        )

def process_photo_for_teacher(selection, selected_date):
    try:
        parts = selection.replace("Пара", "").replace("Аудитория", "").split(",")
        pair_num = parts[0].strip()
        auditorium = parts[1].strip()
    except Exception as e:
        return None, f"❌ Некорректный выбор пары: {str(e)}"

    image, summary = process_photo(auditorium, selected_date, pair_num)

    info = schedule.get(pair_num, {}).get(auditorium)
    if info:
        group = info.get("group", "—")
        discipline = info.get("discipline", "—")
        teacher = info.get("teacher", "—")
        capacity = info.get("capacity", 0)

        details = (
            f"📚 Дисциплина: {discipline}\n"
            f"👥 Группа: {group}\n"
            f"👨‍🏫 Преподаватель: {teacher}\n"
            f"🏫 Аудитория: {auditorium}\n"
            f"🔢 Пара: {pair_num}\n"
        )
        summary = f"{details}\n{summary}"

    return image, summary

def assign_uploaded_photo(uploaded_image, selected_date, pair_num, auditorium):
    if uploaded_image is None:
        return None, "❌ Изображение не загружено."

    try:
        d = date.fromisoformat(selected_date)
        date_str = d.strftime("%d.%m.%Y")
    except ValueError:
        return None, "❌ Неверный формат даты. Используйте YYYY-MM-DD."

    # Проверка, есть ли такая пара в расписании
    if pair_num not in schedule or auditorium not in schedule[pair_num]:
        return None, "❌ В расписании нет такой пары и аудитории."

    # Сохранение
    image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    save_dir = "photos"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"photo{auditorium}_{date_str}({pair_num}).jpg"
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, image)

    info = schedule[pair_num][auditorium]
    summary = (
        f"📅 Дата: {date_str}\n"
        f"🔢 Пара: {pair_num}\n"
        f"🏫 Аудитория: {auditorium}\n"
        f"👨‍🏫 Преподаватель: {info['teacher']}\n"
        f"👥 Группа: {info['group']}\n"
        f"📚 Дисциплина: {info['discipline']}\n"
        f"💾 Фото сохранено как: {save_name}"
    )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), summary

def process_uploaded_image(uploaded_image, selected_date, auditorium, teacher, group, discipline):
    if uploaded_image is None:
        return None, "❌ Изображение не загружено."

    # Проверка даты
    try:
        d = date.fromisoformat(selected_date)
        date_str = d.strftime("%d.%m.%Y")
    except ValueError:
        return None, "❌ Неверный формат даты. Используйте YYYY-MM-DD."

    # Обработка изображения
    image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    enhanced = enhance_image(image)
    results = model(enhanced)

    person_count = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(enhanced, "Person", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Сохранение фото в uploads (если нужно)
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{save_dir}/upload_{auditorium}_{date_str.replace('.', '-')}.jpg"
    cv2.imwrite(save_name, enhanced)

    # Текстовая сводка
    summary = (
        f"📅 Дата: {date_str}\n"
        f"🏫 Аудитория: {auditorium}\n"
        f"👨‍🏫 Преподаватель: {teacher}\n"
        f"👥 Группа: {group}\n"
        f"📚 Дисциплина: {discipline}\n"
        f"🧍 Найдено людей: {person_count}\n"
        f"💾 Сохранено: {save_name}"
    )

    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), summary

def get_all_pairs_choices():
    allowed_pairs = []
    for pair_num, auds in schedule.items():
        for aud_num in auds.keys():
            allowed_pairs.append((pair_num, aud_num))
    choices = [f"Пара {p}, Аудитория {a}" for p, a in allowed_pairs]
    return allowed_pairs, choices

def generate_daily_report(teacher_name, selected_date):
    try:
        d = date.fromisoformat(selected_date)
        date_str = d.strftime("%d.%m.%Y")
    except ValueError:
        return None, "❌ Неверный формат даты. Используйте YYYY-MM-DD."

    report_rows = []
    for pair_num, auds in schedule.items():
        for auditorium, info in auds.items():
            if info["teacher"] == teacher_name:
                filename = f"photo{auditorium}_{date_str}({pair_num}).jpg"
                photo_path = os.path.join("photos", filename)

                count = "-"
                if os.path.exists(photo_path):
                    image = cv2.imread(photo_path)
                    if image is not None:
                        enhanced = enhance_image(image)
                        results = model(enhanced)
                        count = sum(1 for r in results for box in r.boxes if int(box.cls[0]) == 0)

                report_rows.append({
                    "Пара": pair_num,
                    "Аудитория": auditorium,
                    "Группа": info.get("group", "—"),
                    "Дисциплина": info.get("discipline", "—"),
                    "Вместимость": info.get("capacity", 0),
                    "Найдено людей": count
                })

    if not report_rows:
        return None, "⚠️ У преподавателя нет занятий на эту дату."

    output_path = f"отчет_{teacher_name.replace(' ', '_')}_{selected_date}.csv"
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    return output_path, f"✅ Отчет сохранен: {output_path}"

def logout():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        None,
        None,
        gr.update(choices=[]),
        gr.update(visible=False),
    )

with gr.Blocks() as demo:
    gr.HTML("""
    <style>
        body, .gradio-container {
            background-color: #121212 !important;  /* темный фон */
            color: #A8C256 !important;            /* салатовый текст */
        }
        .gr-button, button, input[type="button"], input[type="submit"] {
            background-color: #3A6B35 !important;  /* темно-зеленый фон кнопок */
            color: #D0E8A6 !important;             /* светло-салатовый текст */
            border: none !important;
        }
        .gr-button:hover, button:hover {
            background-color: #5A8C56 !important;  /* светлее на ховере */
        }
        .gr-textbox, .gr-dropdown, .gr-radio, .gr-image {
            background-color: #222 !important;    /* темный фон полей */
            color: #A8C256 !important;             /* салатовый текст */
            border: 1px solid #3A6B35 !important;
        }
        .gr-textbox input, .gr-dropdown select {
            background-color: #222 !important;
            color: #A8C256 !important;
        }
        /* Подписи, лейблы */
        label, .gr-label {
            color: #A8C256 !important;
        }
        /* Заголовки Markdown */
        .gr-markdown {
            color: #B5D68D !important;
        }
    </style>
    """)

    # остальной интерфейс


    # --- Страница логина ---
    with gr.Column():
        login_container = gr.Column(visible=True)
        gr.Markdown("## Вход преподавателя")
        login_input = gr.Textbox(label="Логин")
        password_input = gr.Textbox(label="Пароль", type="password")
        login_btn = gr.Button("Войти")
        login_status = gr.Textbox(label="Статус", interactive=False, visible=True)

    # --- Основной интерфейс (скрыт пока не вошли) ---
    with gr.Column(visible=False) as main_container:
        gr.Markdown("## Основное меню")
        teacher_name_state = gr.Textbox(label="Имя преподавателя", interactive=False, visible=False)
        pairs_dropdown = gr.Dropdown(label="Выберите пару и аудиторию", choices=[])
        date_input = gr.Textbox(label="Дата (ГГГГ-ММ-ДД)", value=str(date.today()))
        process_btn = gr.Button("Показать фото и данные")
        photo_output = gr.Image()
        summary_output = gr.Textbox(label="Информация", interactive=False, lines=7)
        gr.Markdown("### Или загрузите изображение вручную")
        upload = gr.Image(label="Загрузите фото", type="pil")
        upload_date = gr.Textbox(label="Дата (ГГГГ-ММ-ДД)", value=str(date.today()))
        upload_auditorium = gr.Dropdown(label="Аудитория", choices=["110", "120", "130", "140"])
        upload_pair = gr.Dropdown(label="Номер пары", choices=list(schedule.keys()))
        upload_button = gr.Button("Сохранить фото к паре")
        upload_output_image = gr.Image()
        upload_output_text = gr.Textbox(label="Результат", interactive=False)

        logout_btn = gr.Button("Выйти")

    with gr.Column(visible=False) as admin_edit_container:
        gr.Markdown("## Редактирование расписания (только для администратора)")
        pair_for_edit = gr.Dropdown(label="Выберите пару", choices=list(schedule.keys()))
        auditorium_for_edit = gr.Dropdown(label="Выберите аудиторию", choices=["110", "120", "130", "140"])
        teacher_edit = gr.Textbox(label="Преподаватель")
        group_edit = gr.Textbox(label="Группа")
        discipline_edit = gr.Textbox(label="Дисциплина")
        capacity_edit = gr.Number(label="Вместимость", value=0)

        save_btn = gr.Button("Сохранить изменения")
        gr.Markdown("## Отчет по всем фото")
        report_date_input = gr.Textbox(label="Дата отчета (ГГГГ-ММ-ДД)", value=str(date.today()))
        generate_report_btn = gr.Button("Сформировать отчет")
        report_output = gr.Textbox(label="Отчет", interactive=False, lines=15)
        save_status = gr.Textbox(label="Статус", interactive=False)


    # Связи и логика
    login_btn.click(
        fn=try_login,
        inputs=[login_input, password_input],
        outputs=[
            login_container,
            main_container,
            login_status,
            teacher_name_state,
            gr.State(),
            pairs_dropdown,
            admin_edit_container
        ]
    )

    # Хранение выбранных пар в состоянии
    pairs_state = gr.State([])

    # Обновление выбранных пар при входе
    def update_pairs_state(teacher_name, pairs):
        return pairs


    teacher_name_state.change(
        fn=update_pairs_state,
        inputs=[teacher_name_state, pairs_dropdown],
        outputs=pairs_state
    )

    # Обработка нажатия показать фото
    def on_process(pairs_dropdown_value, selected_date):
        return process_photo_for_teacher(pairs_dropdown_value, selected_date)

    process_btn.click(
        fn=on_process,
        inputs=[pairs_dropdown, date_input],
        outputs=[photo_output, summary_output]
    )
    pair_for_edit.change(
        fn=load_pair_data,
        inputs=[pair_for_edit, auditorium_for_edit],
        outputs=[teacher_edit, group_edit, discipline_edit, capacity_edit]
    )

    auditorium_for_edit.change(
        fn=load_pair_data,
        inputs=[pair_for_edit, auditorium_for_edit],
        outputs=[teacher_edit, group_edit, discipline_edit, capacity_edit]
    )

    save_btn.click(
        fn=save_schedule_change,
        inputs=[pair_for_edit, auditorium_for_edit, teacher_edit, group_edit, discipline_edit, capacity_edit],
        outputs=[save_status]
    )
    upload_button.click(
        fn=assign_uploaded_photo,
        inputs=[upload, upload_date, upload_pair, upload_auditorium],
        outputs=[upload_output_image, upload_output_text]
    )
    generate_report_btn.click(
        fn=generate_admin_report,
        inputs=[report_date_input],
        outputs=[report_output]
    )

    # Кнопка выхода
    save_btn.click(
        fn=save_schedule_change,
        inputs=[pair_for_edit, auditorium_for_edit, teacher_edit, group_edit, discipline_edit, capacity_edit],
        outputs=[save_status, gr.State(), pairs_dropdown]
    )

demo.launch(share=True)





