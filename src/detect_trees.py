"""
Tree Detection Module for Aerial Imagery
========================================
Детекция деревьев на аэрофотоснимках с использованием DeepForest.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from deepforest import main

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TreeDetector:
    """Детектор деревьев на основе DeepForest."""
    
    def __init__(
        self,
        model_name: str = "weecology/deepforest-tree",
        device: Optional[str] = None,
        confidence_threshold: float = 0.4,
        patch_size: int = 400,
        patch_overlap: int = 0
    ):
        self.model_name = model_name
        self.device = device or self._get_device()
        self.confidence_threshold = confidence_threshold
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.model = None
        
        logger.info(f"Устройство: {self.device}")
        logger.info(f"Порог уверенности: {self.confidence_threshold}")
    
    def _get_device(self) -> str:
        """Определяет доступное устройство."""
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        logger.info("GPU не найден, используется CPU")
        return 'cpu'
    
    def load(self) -> 'TreeDetector':
        """Загружает модель DeepForest."""
        try:
            logger.info(f"Загрузка модели: {self.model_name}")
            self.model = main.deepforest()
            self.model.load_model(self.model_name)
            logger.info("Модель загружена")
        except Exception as e:
            logger.warning(f"Ошибка загрузки: {e}")
            logger.info("Загрузка стандартной модели...")
            self.model = main.deepforest()
            self.model.use_release()
            logger.info("Стандартная модель загружена")
        
        # Перенос на устройство
        if hasattr(self.model, 'model'):
            self.model.model.to(self.device)
            self.model.model.eval()
        
        return self
    
    def predict(self, image_path: Union[str, Path]) -> pd.DataFrame:
        """Выполняет детекцию деревьев на изображении."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        
        logger.info(f"Обработка: {image_path.name}")
        
        # Проверяем, что модель загружена
        if self.model is None:
            raise RuntimeError("Модель не загружена. Вызовите .load() сначала.")
        
        # Выполняем предсказание
        try:
            detections = self.model.predict_tile(
                str(image_path),
                patch_size=self.patch_size,
                patch_overlap=self.patch_overlap
            )
        except AttributeError:
            # Если predict_tile не работает, пробуем predict_image
            logger.info("Использование predict_image вместо predict_tile")
            detections = self.model.predict_image(str(image_path))
        
        # Проверяем, что detections - это DataFrame
        if detections is None or len(detections) == 0:
            logger.warning("Деревья не обнаружены")
            return pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'score'])
        
        # Фильтрация по уверенности
        if 'score' in detections.columns:
            initial_count = len(detections)
            detections = detections[detections['score'] >= self.confidence_threshold]
        
        logger.info(f"Обнаружено деревьев: {len(detections)}")
        return detections
    
    def visualize(
        self,
        image_path: Union[str, Path],
        detections: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        show_labels: bool = True,
        color: str = '#00ff00'
    ) -> plt.Figure:
        """Визуализирует результаты детекции."""
        # Загружаем изображение
        try:
            image = plt.imread(image_path)
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения: {e}")
            # Пробуем через PIL
            with Image.open(image_path) as img:
                image = np.array(img)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image)
        
        # Рисуем рамки
        if len(detections) > 0:
            for _, row in detections.iterrows():
                # Проверяем наличие всех координат
                if all(k in row for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                    rect = Rectangle(
                        (row['xmin'], row['ymin']),
                        row['xmax'] - row['xmin'],
                        row['ymax'] - row['ymin'],
                        fill=False,
                        edgecolor=color,
                        linewidth=2.5,
                        alpha=0.9
                    )
                    ax.add_patch(rect)
                    
                    # Метка уверенности
                    if show_labels and 'score' in row:
                        ax.text(
                            row['xmin'],
                            row['ymin'] - 10,
                            f"{row['score']:.2f}",
                            color='white',
                            fontsize=9,
                            fontweight='bold',
                            bbox=dict(
                                facecolor='black',
                                alpha=0.7,
                                pad=3,
                                boxstyle='round,pad=0.3'
                            )
                        )
        
        ax.set_title(f"Обнаружено деревьев: {len(detections)}", fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        # Сохранение
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Сохранено: {save_path}")
        
        return fig
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path] = "results"
    ) -> pd.DataFrame:
        """Обрабатывает все изображения в директории."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Собираем все изображения
        extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        image_files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        if not image_files:
            logger.warning(f"Изображения не найдены в: {input_dir}")
            return pd.DataFrame()
        
        logger.info(f"Найдено файлов: {len(image_files)}")
        
        results = []
        for img_path in image_files:
            try:
                logger.info(f"Обработка: {img_path.name}")
                
                # Детекция
                detections = self.predict(str(img_path))
                
                # Сохраняем CSV
                if len(detections) > 0:
                    csv_path = output_dir / f"{img_path.stem}_detections.csv"
                    detections.to_csv(csv_path, index=False)
                
                # Визуализация
                viz_path = output_dir / f"{img_path.stem}_result.png"
                self.visualize(str(img_path), detections, save_path=viz_path)
                
                # Статистика
                results.append({
                    'filename': img_path.name,
                    'trees_detected': len(detections),
                    'avg_confidence': detections['score'].mean() if len(detections) > 0 and 'score' in detections.columns else 0
                })
                
            except Exception as e:
                logger.error(f"Ошибка {img_path.name}: {e}")
                results.append({
                    'filename': img_path.name,
                    'error': str(e),
                    'trees_detected': 0
                })
        
        # Сводка
        summary = pd.DataFrame(results)
        summary_path = output_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"Сводка сохранена: {summary_path}")
        
        total_trees = summary['trees_detected'].sum()
        logger.info(f"Всего обнаружено деревьев: {total_trees}")
        
        return summary


def main():
    """Точка входа из командной строки."""
    parser = argparse.ArgumentParser(
        description="Детекция деревьев на аэрофотоснимках"
    )
    parser.add_argument("input", help="Путь к изображению или директории")
    parser.add_argument("--output", default="results", help="Директория для результатов")
    parser.add_argument("--threshold", type=float, default=0.4, help="Порог уверенности (0-1)")
    parser.add_argument("--patch-size", type=int, default=400, help="Размер патча")
    parser.add_argument("--batch", action="store_true", help="Пакетный режим")
    parser.add_argument("--no-labels", action="store_true", help="Не показывать метки")
    parser.add_argument("--color", default="#00ff00", help="Цвет рамок")
    
    args = parser.parse_args()
    
    # Инициализация
    detector = TreeDetector(
        confidence_threshold=args.threshold,
        patch_size=args.patch_size
    ).load()
    
    input_path = Path(args.input)
    
    try:
        if args.batch or input_path.is_dir():
            # Пакетная обработка
            detector.process_directory(input_path, args.output)
        else:
            # Одиночный файл
            detections = detector.predict(str(input_path))
            
            # Сохранение
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if len(detections) > 0:
                csv_path = output_dir / f"{input_path.stem}_detections.csv"
                detections.to_csv(csv_path, index=False)
            
            viz_path = output_dir / f"{input_path.stem}_result.png"
            detector.visualize(
                str(input_path),
                detections,
                save_path=viz_path,
                show_labels=not args.no_labels,
                color=args.color
            )
            
            logger.info(f"✅ Готово! Результаты в: {output_dir}")
            
    except KeyboardInterrupt:
        logger.info("Операция прервана")
        return 1
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())