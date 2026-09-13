# Tree Detection

Инструмент для детекции деревьев на аэрофотоснимках. Использует предобученную 
модель DeepForest, которая определяет координаты деревьев (bounding boxes) 
и сохраняет результаты в CSV.

## Зачем это нужно

Скрипт писался для задачи подсчёта деревьев на лесных снимках. Модель 
DeepForest показала хорошие результаты, поэтому обернул её в удобный класс 
с визуализацией и пакетной обработкой.

## Что умеет

- Обрабатывать одно изображение или целую папку
- Рисовать рамки вокруг найденных деревьев и сохранять результат
- Сохранять координаты и уверенность модели в CSV
- Работать на CPU и GPU
- Позволять настраивать порог уверенности

## Установка

Требуется Python 3.8+.

Создаём виртуальное окружение:

    python -m venv .venv

Активируем:
    
    .venv\Scripts\activate      # Windows
    source .venv/bin/activate   # Linux/Mac

Обновляем pip и ставим зависимости:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

Если ставите вручную, порядок важен. Сначала torch, потом остальное:

    pip install torch torchvision
    pip install numpy pandas matplotlib Pillow opencv-python
    pip install shapely geopandas
    pip install deepforest

С geopandas на Windows бывают проблемы — лучше ставить через conda, 
если он есть:

    conda install -c conda-forge geopandas

## Использование

Одно изображение:

    python tree_detector.py image.jpg

Результат появится в папке results/ — PNG с разметкой и CSV с координатами.

Указать другую папку:

    python tree_detector.py image.jpg --output my_results/

Обработать всю папку:

    python tree_detector.py ./images/ --batch

Поменять порог уверенности и цвет рамок:

    python tree_detector.py image.jpg --threshold 0.6 --color red

Все параметры:

    input                 путь к файлу или папке (обязательно)
    --output              куда сохранять (по умолчанию results)
    --threshold           порог уверенности 0-1 (по умолчанию 0.4)
    --patch-size          размер патча (по умолчанию 400)
    --patch-overlap       перекрытие патчей (по умолчанию 0)
    --batch               пакетная обработка папки
    --no-labels           не рисовать метки уверенности
    --color               цвет рамок (по умолчанию #00ff00)
    --device              cuda или cpu (по умолчанию авто)

## Что на выходе

Для каждого изображения создаётся два файла:

    results/
      image_result.png        — картинка с рамками
      image_detections.csv    — координаты и уверенность

Формат CSV:

    xmin, ymin, xmax, ymax, score, label
    120,  340,  145,  365,  0.89,  Tree
    200,  410,  230,  440,  0.92,  Tree

В пакетном режиме дополнительно создаётся summary.csv со статистикой 
по всем файлам.

## Структура проекта

    tree-detector/
      tree_detector.py     — основной код
      requirements.txt     — зависимости
      README.md
      .gitignore
      data/                — сюда класть изображения
      results/             — сюда пишутся результаты

## Зависимости

    torch, torchvision   — фреймворк и утилиты
    deepforest           — сама модель детекции
    numpy, pandas        — данные
    matplotlib, Pillow   — визуализация
    opencv-python        — обработка изображений
    shapely, geopandas   — геометрия

## Известные проблемы

- На Windows установка geopandas через pip часто падает из-за GDAL. 
  Решение — conda или предварительная установка GDAL.
- Первый запуск долгий, потому что модель скачивается с HuggingFace 
  (~200 МБ). Дальше берётся из кэша.
- Если нет GPU, работает на CPU, но медленнее. Для больших снимков 
  лучше GPU.

## Лицензия

MIT. Делайте что хотите.

## Автор

Obid-Jon
https://github.com/Obid-Jon