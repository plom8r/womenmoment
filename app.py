from fastapi import FastAPI
import gradio as gr
from yolomini import enhance_image, generate_admin_report, process_photo, get_teacher_pairs, load_pair_data, save_schedule_change, try_login, process_photo_for_teacher, assign_uploaded_photo, process_uploaded_image, get_all_pairs_choices, generate_daily_report, logout  # импорт своих функций

app = FastAPI()

# 1. Gradio-интерфейс
def gradio_ui():
    with gr.Blocks() as demo:
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
    return demo

demo = gradio_ui()

# 2. Встраиваем Gradio в FastAPI
app = gr.mount_gradio_app(app, demo, path="/gradio")

@app.get("/")
async def root():
    return {"message": "Сервер работает. Перейди по /gradio для интерфейса."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
