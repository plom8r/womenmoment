from ultralytics import YOLO
import os

def train_yolo_for_human_silhouettes():

    model = YOLO("runs/detect/yolo_person_only/weights/best.pt")

    # Настройка обучения только для класса 'person' (id=0 в COCO)
    model.model.args['classes'] = [0]  # Только класс 'person'

    # Параметры обучения
    train_args = results = model.train {
        data="coco_person.yaml",  # Наш конфиг
        epochs=50,               # Количество эпох
        batch=16,                # Размер батча
        imgsz=640,               # Размер изображения
        device="0",              # GPU (или "cpu" если нет видеокарты)
        name="yolo_person_only", # Название эксперимента
        single_cls=True,         # Обучаем как одноклассовую модель
    }

    # Запуск обучения
    results = model.train(**train_args)

    # Сохранение лучшей модели
    best_model_path = os.path.join(train_args['name'], 'weights/best.pt')
    return best_model_path

if __name__ == '__main__':
    trained_model_path = train_yolo_for_human_silhouettes()
    print(f"Модель сохранена в: {trained_model_path}")