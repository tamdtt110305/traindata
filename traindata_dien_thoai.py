import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import warnings
import json
import time
from pathlib import Path
import logging
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
from PIL import Image

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tắt warnings không cần thiết
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EnhancedPhoneDetectionCNN:
    def __init__(self, img_height=224, img_width=224, use_pretrained=True):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.class_names = ['no_phone', 'using_phone']
        self.training_config = {}
        self.history = None
        self.use_pretrained = use_pretrained
        
        # Kiểm tra GPU
        self._check_gpu()
        
        # Thiết lập memory growth cho GPU
        self._setup_gpu_memory()
    
    def _check_gpu(self):
        """Kiểm tra và thiết lập GPU"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logger.info(f"🎮 Tìm thấy {len(gpus)} GPU(s)")
            for gpu in gpus:
                logger.info(f"   - {gpu}")
        else:
            logger.warning("⚠️  Không tìm thấy GPU, sử dụng CPU")
    
    def _setup_gpu_memory(self):
        """Thiết lập GPU memory growth để tránh lỗi OOM"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("✅ Đã thiết lập GPU memory growth")
        except RuntimeError as e:
            logger.warning(f"⚠️  Không thể thiết lập GPU memory growth: {e}")
    
    def _normalize_path(self, path):
        """Chuẩn hóa đường dẫn để tránh lỗi backslash trên Windows"""
        if path:
            return str(Path(path).resolve())
        return path
    
    def _validate_data_structure(self, data_dir):
        """Kiểm tra cấu trúc dữ liệu với xử lý đường dẫn cải thiện"""
        data_dir = self._normalize_path(data_dir)
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ Không tìm thấy thư mục: {data_dir}")
        
        class_counts = {}
        total_images = 0
        valid_files = []
        
        for class_name in self.class_names:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                logger.warning(f"⚠️  Không tìm thấy thư mục class: {class_path}")
                class_counts[class_name] = 0
                continue
            
            # Đếm số ảnh trong class và kiểm tra file
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            class_files = []
            
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        full_path = os.path.join(root, file)
                        # Kiểm tra file có thể đọc được không
                        try:
                            if os.path.getsize(full_path) > 0:  # File không rỗng
                                # Thêm validation cho ảnh
                                if self._is_valid_image(full_path):
                                    class_files.append(full_path)
                                    valid_files.append(full_path)
                        except (OSError, IOError) as e:
                            logger.warning(f"⚠️  File lỗi: {full_path} - {e}")
            
            class_counts[class_name] = len(class_files)
            total_images += len(class_files)
            
            logger.info(f"   {class_name}: {len(class_files)} ảnh")
        
        if total_images == 0:
            raise ValueError(f"❌ Không tìm thấy ảnh hợp lệ nào trong {data_dir}")
        
        # Kiểm tra class imbalance
        if len(set(class_counts.values())) > 1:
            ratio = max(class_counts.values()) / max(min(class_counts.values()), 1)
            if ratio > 3:
                logger.warning(f"⚠️  Dữ liệu không cân bằng (tỷ lệ: {ratio:.1f}:1)")
                logger.info("💡 Khuyến nghị sử dụng class weights hoặc augmentation")
        
        return class_counts, total_images, valid_files
    
    def _is_valid_image(self, image_path):
        """Kiểm tra tính hợp lệ của ảnh"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def build_model(self, num_classes=2, dropout_rate=0.3):
        """Tạo model với Transfer Learning hoặc CNN custom tùy chọn"""
        try:
            if self.use_pretrained:
                # Sử dụng Transfer Learning với MobileNetV2
                logger.info("🔄 Sử dụng Transfer Learning với MobileNetV2...")
                
                # Base model
                base_model = MobileNetV2(
                    input_shape=(self.img_height, self.img_width, 3),
                    include_top=False,
                    weights='imagenet'
                )
                
                # Freeze base model layers
                base_model.trainable = False
                
                # Add custom head
                model = models.Sequential([
                    layers.Input(shape=(self.img_height, self.img_width, 3)),
                    layers.Rescaling(1./255),
                    
                    # Preprocessing layers for better augmentation
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.1),
                    layers.RandomZoom(0.1),
                    layers.RandomContrast(0.1),
                    layers.RandomBrightness(0.1),
                    
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.BatchNormalization(),
                    layers.Dropout(dropout_rate),
                    layers.Dense(128, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(dropout_rate * 0.5),
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(dropout_rate * 0.3),
                    layers.Dense(num_classes, activation='softmax')
                ])
                
            else:
                # Custom CNN với kiến trúc tối ưu hơn
                logger.info("🏗️ Xây dựng Custom CNN...")
                
                model = models.Sequential([
                    layers.Input(shape=(self.img_height, self.img_width, 3)),
                    layers.Rescaling(1./255),
                    
                    # Augmentation layers
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.1),
                    layers.RandomZoom(0.1),
                    layers.RandomContrast(0.1),
                    layers.RandomBrightness(0.1),
                    
                    # Block 1
                    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                    layers.BatchNormalization(),
                    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Dropout(0.25),
                    
                    # Block 2
                    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                    layers.BatchNormalization(),
                    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Dropout(0.25),
                    
                    # Block 3
                    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                    layers.BatchNormalization(),
                    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Dropout(0.25),
                    
                    # Block 4
                    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                    layers.BatchNormalization(),
                    layers.GlobalAveragePooling2D(),
                    
                    # Classification head
                    layers.Dense(512, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(dropout_rate),
                    layers.Dense(256, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(dropout_rate * 0.5),
                    layers.Dense(num_classes, activation='softmax')
                ]
                )
            
            self.model = model
            logger.info(f"✅ Model tạo thành công với {model.count_params():,} parameters")
            
            # In model summary
            model.summary()
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo model: {e}")
            raise
    
    def create_data_generators(self, train_dir, val_dir=None, batch_size=32, 
                             validation_split=0.2):
        """Tạo data generators với augmentation mạnh hơn"""
        try:
            # Chuẩn hóa đường dẫn
            train_dir = self._normalize_path(train_dir)
            if val_dir:
                val_dir = self._normalize_path(val_dir)
            
            # Validate dữ liệu
            logger.info("📊 Kiểm tra dữ liệu training...")
            train_counts, train_total, valid_files = self._validate_data_structure(train_dir)
            
            # Tạo validation từ train nếu không có val_dir
            if val_dir is None or not os.path.exists(val_dir):
                logger.info(f"📄 Tạo validation split từ training data ({validation_split*100}%)")
                
                # Training generator với augmentation nhẹ hơn
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=validation_split,
                    rotation_range=8,           # giảm từ 20
                    width_shift_range=0.08,     # giảm từ 0.2
                    height_shift_range=0.08,    # giảm từ 0.2
                    shear_range=0.05,           # giảm từ 0.15
                    zoom_range=0.08,            # giảm từ 0.2
                    horizontal_flip=True,
                    brightness_range=[0.9, 1.1],# giảm biên độ
                    channel_shift_range=0.05,   # giảm từ 0.1
                    fill_mode='nearest'
                )
                
                # Validation generator không augmentation
                val_datagen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=validation_split
                )
                
                train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(self.img_height, self.img_width),
                    batch_size=batch_size,
                    class_mode='categorical',
                    classes=self.class_names,
                    subset='training',
                    shuffle=True,
                    seed=42,
                    interpolation='bilinear'
                )
                
                val_generator = val_datagen.flow_from_directory(
                    train_dir,
                    target_size=(self.img_height, self.img_width),
                    batch_size=batch_size,
                    class_mode='categorical',
                    classes=self.class_names,
                    subset='validation',
                    shuffle=False,
                    seed=42,
                    interpolation='bilinear'
                )
                
            else:
                logger.info("📊 Kiểm tra dữ liệu validation...")
                val_counts, val_total, _ = self._validate_data_structure(val_dir)
                
                # Separate train và validation directories
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=8,
                    width_shift_range=0.08,
                    height_shift_range=0.08,
                    shear_range=0.05,
                    zoom_range=0.08,
                    horizontal_flip=True,
                    brightness_range=[0.9, 1.1],
                    channel_shift_range=0.05,
                    fill_mode='nearest'
                )
                
                val_datagen = ImageDataGenerator(rescale=1./255)
                
                train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(self.img_height, self.img_width),
                    batch_size=batch_size,
                    class_mode='categorical',
                    classes=self.class_names,
                    shuffle=True,
                    seed=42,
                    interpolation='bilinear'
                )
                
                val_generator = val_datagen.flow_from_directory(
                    val_dir,
                    target_size=(self.img_height, self.img_width),
                    batch_size=batch_size,
                    class_mode='categorical',
                    classes=self.class_names,
                    shuffle=False,
                    seed=42,
                    interpolation='bilinear'
                )
            
            # Kiểm tra batch đầu tiên
            self._test_generators(train_generator, val_generator)
            
            return train_generator, val_generator
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo data generators: {e}")
            raise
    
    def _test_generators(self, train_gen, val_gen):
        """Test generators để đảm bảo hoạt động bình thường"""
        try:
            logger.info("🧪 Test data generators...")
            
            # Test train generator
            train_batch = next(train_gen)
            logger.info(f"   Train batch shape: {train_batch[0].shape}")
            logger.info(f"   Train labels shape: {train_batch[1].shape}")
            logger.info(f"   Train data range: [{train_batch[0].min():.3f}, {train_batch[0].max():.3f}]")
            
            # Test validation generator
            if val_gen:
                val_batch = next(val_gen)
                logger.info(f"   Val batch shape: {val_batch[0].shape}")
                logger.info(f"   Val labels shape: {val_batch[1].shape}")
                logger.info(f"   Val data range: [{val_batch[0].min():.3f}, {val_batch[0].max():.3f}]")
            
            logger.info("✅ Data generators hoạt động bình thường")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi test generators: {e}")
            raise
    
    def calculate_class_weights(self, train_generator):
        """Tính class weights để xử lý imbalanced data"""
        try:
            # Lấy labels từ generator
            labels = train_generator.classes
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(labels),
                y=labels
            )
            
            class_weight_dict = dict(enumerate(class_weights))
            logger.info(f"📊 Class weights: {class_weight_dict}")
            
            return class_weight_dict
            
        except Exception as e:
            logger.warning(f"⚠️  Không thể tính class weights: {e}")
            return None
    
    def train(self, train_generator, val_generator=None, epochs=50, 
              learning_rate=0.001, save_path='models/enhanced_phone_detection_model.keras',
              use_class_weights=True, patience=15, fine_tune_epochs=10):
        """Training với Transfer Learning và Fine-tuning"""
        try:
            logger.info("🚀 Bắt đầu training...")
            
            # Tạo thư mục lưu model
            save_path = self._normalize_path(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Lưu config
            self.training_config = {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': train_generator.batch_size,
                'image_size': (self.img_height, self.img_width),
                'total_train_samples': train_generator.samples,
                'total_val_samples': val_generator.samples if val_generator else 0,
                'use_class_weights': use_class_weights,
                'use_pretrained': self.use_pretrained,
                'fine_tune_epochs': fine_tune_epochs
            }
            
            # Tính class weights
            class_weights = None
            if use_class_weights:
                class_weights = self.calculate_class_weights(train_generator)
            
            # Phase 1: Train with frozen base model
            logger.info("📚 Phase 1: Training với base model frozen...")
            
            optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )
            
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )
            
            # Callbacks
            callbacks = self._create_callbacks(save_path, val_generator, patience)
            
            # Tính steps per epoch
            steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
            validation_steps = None
            if val_generator:
                validation_steps = max(1, val_generator.samples // val_generator.batch_size)
            
            logger.info(f"📊 Training info:")
            logger.info(f"   Steps per epoch: {steps_per_epoch}")
            logger.info(f"   Validation steps: {validation_steps}")
            logger.info(f"   Total parameters: {self.model.count_params():,}")
            logger.info(f"   Learning rate: {learning_rate}")
            
            # Phase 1 Training
            start_time = time.time()
            
            history1 = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Phase 2: Fine-tuning (nếu sử dụng pretrained model)
            if self.use_pretrained and fine_tune_epochs > 0:
                logger.info("🔧 Phase 2: Fine-tuning...")
                
                # Unfreeze top layers của base model
                base_model = self.model.layers[6]  # MobileNetV2 layer
                base_model.trainable = True
                
                # Freeze bottom layers, chỉ train top layers
                for layer in base_model.layers[:-20]:
                    layer.trainable = False
                
                # Compile với learning rate thấp hơn
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate/10),
                    loss='categorical_crossentropy',
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                    ]
                )
                
                # Fine-tuning callbacks
                fine_tune_callbacks = self._create_callbacks(
                    save_path.replace('.keras', '_fine_tuned.keras'), 
                    val_generator, 
                    patience//2
                )
                
                history2 = self.model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=fine_tune_epochs,
                    validation_data=val_generator,
                    validation_steps=validation_steps,
                    callbacks=fine_tune_callbacks,
                    class_weight=class_weights,
                    verbose=1
                )
                
                # Combine histories
                history = self._combine_histories(history1, history2)
            else:
                history = history1
            
            training_time = time.time() - start_time
            logger.info(f"⏱️ Training hoàn thành trong {training_time/60:.1f} phút")
            
            # Lưu history và config
            self.history = history
            self._save_training_info(save_path, history, training_time)
            
            # Vẽ kết quả
            self.plot_training_history(history, save_path)
            
            return history
            
        except KeyboardInterrupt:
            logger.info("⏹️ Training bị dừng bởi người dùng")
            if self.model:
                interrupt_path = save_path.replace('.keras', '_interrupted.keras')
                self.model.save(interrupt_path)
                logger.info(f"💾 Model đã lưu tại: {interrupt_path}")
            raise
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình training: {e}")
            raise
    
    def _combine_histories(self, history1, history2):
        """Kết hợp 2 history objects"""
        combined = type('History', (), {})()
        combined.history = {}
        
        for key in history1.history:
            combined.history[key] = history1.history[key] + history2.history[key]
        
        return combined
    
    def _create_callbacks(self, save_path, val_generator, patience):
        """Tạo callbacks cải thiện"""
        callbacks = []
        
        # Early stopping
        monitor = 'val_loss' if val_generator else 'loss'
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min',
            min_delta=0.0001
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            save_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        )
        callbacks.append(checkpoint)
        
        # Reduce learning rate
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(5, patience//3),
            min_lr=1e-8,
            verbose=1,
            mode='min',
            min_delta=0.0001
        )
        callbacks.append(reduce_lr)
        
        # CSV Logger
        csv_path = save_path.replace('.keras', '_training_log.csv')
        csv_logger = keras.callbacks.CSVLogger(csv_path, append=True)
        callbacks.append(csv_logger)
        
        return callbacks
    
    def _save_training_info(self, save_path, history, training_time):
        """Lưu thông tin training"""
        try:
            info_path = save_path.replace('.keras', '_info.json')
            
            # Chuẩn bị thông tin
            training_info = {
                'config': self.training_config,
                'training_time_minutes': training_time / 60,
                'final_metrics': {
                    'train_loss': float(history.history['loss'][-1]),
                    'train_accuracy': float(history.history['accuracy'][-1]),
                },
                'best_metrics': {
                    'best_train_loss': float(min(history.history['loss'])),
                    'best_train_accuracy': float(max(history.history['accuracy'])),
                },
                'total_epochs': len(history.history['loss'])
            }
            
            # Thêm validation metrics nếu có
            if 'val_loss' in history.history:
                training_info['final_metrics'].update({
                    'val_loss': float(history.history['val_loss'][-1]),
                    'val_accuracy': float(history.history['val_accuracy'][-1]),
                })
                training_info['best_metrics'].update({
                    'best_val_loss': float(min(history.history['val_loss'])),
                    'best_val_accuracy': float(max(history.history['val_accuracy'])),
                })
            
            # Lưu file
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(training_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 Training info saved: {info_path}")
            
        except Exception as e:
            logger.warning(f"⚠️  Không thể lưu training info: {e}")
    
    def plot_training_history(self, history, save_path):
        """Vẽ biểu đồ training history cải thiện"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            axes[0,0].plot(history.history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in history.history:
                axes[0,0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0,0].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Loss')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Accuracy plot
            axes[0,1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
            if 'val_accuracy' in history.history:
                axes[0,1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[0,1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('Accuracy')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Precision plot
            if 'precision' in history.history:
                axes[1,0].plot(history.history['precision'], label='Training Precision', linewidth=2)
                if 'val_precision' in history.history:
                    axes[1,0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
                axes[1,0].set_title('Model Precision', fontsize=14, fontweight='bold')
                axes[1,0].set_xlabel('Epoch')
                axes[1,0].set_ylabel('Precision')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
            
            # F1 Score plot
            if 'f1_score' in history.history:
                axes[1,1].plot(history.history['f1_score'], label='Training F1 Score', linewidth=2)
                if 'val_f1_score' in history.history:
                    axes[1,1].plot(history.history['val_f1_score'], label='Validation F1 Score', linewidth=2)
                axes[1,1].set_title('Model F1 Score', fontsize=14, fontweight='bold')
                axes[1,1].set_xlabel('Epoch')
                axes[1,1].set_ylabel('F1 Score')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Lưu biểu đồ
            plot_path = save_path.replace('.keras', '_training_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"📈 Training plot saved: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"⚠️  Không thể vẽ biểu đồ: {e}")
    
    def evaluate_model(self, test_generator, save_results=True, save_path=None):
        """Đánh giá model chi tiết với confusion matrix"""
        try:
            if self.model is None:
                raise ValueError("Model chưa được training hoặc load!")
            
            logger.info("🧪 Đánh giá model...")
            
            # Reset generator
            test_generator.reset()
            
            # Predict
            steps = max(1, test_generator.samples // test_generator.batch_size)
            predictions = self.model.predict(test_generator, steps=steps, verbose=1)
            
            # Get true labels
            true_labels = test_generator.classes[:len(predictions)]
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, predicted_labels, average='weighted'
            )
            
            logger.info(f"📊 Kết quả đánh giá:")
            logger.info(f"   Accuracy: {accuracy:.4f}")
            logger.info(f"   Precision: {precision:.4f}")
            logger.info(f"   Recall: {recall:.4f}")
            logger.info(f"   F1 Score: {f1:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            
            if save_results and save_path:
                self._plot_confusion_matrix(cm, save_path)
                self._save_classification_report(true_labels, predicted_labels, save_path)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': predictions,
                'true_labels': true_labels
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi đánh giá model: {e}")
            raise
    
    def _plot_confusion_matrix(self, cm, save_path):
        """Vẽ confusion matrix"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            
            cm_path = save_path.replace('.keras', '_confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"📊 Confusion matrix saved: {cm_path}")
            plt.show()
            
        except Exception as e:
            logger.warning(f"⚠️  Không thể vẽ confusion matrix: {e}")
    
    def _save_classification_report(self, true_labels, predicted_labels, save_path):
        """Lưu classification report"""
        try:
            report = classification_report(
                true_labels, predicted_labels,
                target_names=self.class_names,
                output_dict=True
            )
            
            report_path = save_path.replace('.keras', '_classification_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Classification report saved: {report_path}")
            
            # In report ra console
            print("\n" + "="*50)
            print("CLASSIFICATION REPORT")
            print("="*50)
            print(classification_report(true_labels, predicted_labels, target_names=self.class_names))
            
        except Exception as e:
            logger.warning(f"⚠️  Không thể lưu classification report: {e}")
    
    def predict_single_image(self, image_path, show_result=True):
        """Dự đoán cho một ảnh"""
        try:
            if self.model is None:
                raise ValueError("Model chưa được training hoặc load!")
            
            # Chuẩn hóa đường dẫn
            image_path = self._normalize_path(image_path)
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
            
            # Load và preprocess ảnh
            img = keras.utils.load_img(
                image_path, 
                target_size=(self.img_height, self.img_width)
            )
            img_array = keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Tạo batch
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            confidence = np.max(predictions[0])
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            
            result = {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'all_predictions': {
                    self.class_names[i]: float(predictions[0][i]) 
                    for i in range(len(self.class_names))
                }
            }
            
            if show_result:
                self._show_prediction_result(image_path, img, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi dự đoán ảnh: {e}")
            raise
    
    def _show_prediction_result(self, image_path, img, result):
        """Hiển thị kết quả dự đoán"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Hiển thị ảnh
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Input Image\n{os.path.basename(image_path)}", fontsize=12)
            plt.axis('off')
            
            # Hiển thị kết quả
            plt.subplot(1, 2, 2)
            classes = list(result['all_predictions'].keys())
            confidences = list(result['all_predictions'].values())
            
            colors = ['red' if c == result['predicted_class'] else 'blue' for c in classes]
            bars = plt.bar(classes, confidences, color=colors, alpha=0.7)
            
            plt.title(f"Prediction Results\nPredicted: {result['predicted_class']}\nConfidence: {result['confidence']:.3f}", fontsize=12)
            plt.ylabel('Confidence')
            plt.ylim(0, 1)
            
            # Thêm text trên bars
            for bar, conf in zip(bars, confidences):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # In kết quả
            print(f"\n🎯 Kết quả dự đoán cho: {os.path.basename(image_path)}")
            print(f"   Predicted Class: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   All Predictions:")
            for class_name, conf in result['all_predictions'].items():
                print(f"     {class_name}: {conf:.4f}")
            
        except Exception as e:
            logger.warning(f"⚠️  Không thể hiển thị kết quả: {e}")
    
    def predict_batch_images(self, image_folder, output_csv=None, min_confidence=0.5):
        """Dự đoán batch ảnh"""
        try:
            if self.model is None:
                raise ValueError("Model chưa được training hoặc load!")
            
            image_folder = self._normalize_path(image_folder)
            
            if not os.path.exists(image_folder):
                raise FileNotFoundError(f"Không tìm thấy thư mục: {image_folder}")
            
            # Tìm tất cả ảnh
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            image_files = []
            
            for root, dirs, files in os.walk(image_folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                raise ValueError(f"Không tìm thấy ảnh nào trong {image_folder}")
            
            logger.info(f"🔍 Đang dự đoán {len(image_files)} ảnh...")
            
            results = []
            failed_images = []
            
            for i, image_path in enumerate(image_files):
                try:
                    result = self.predict_single_image(image_path, show_result=False)
                    result['image_path'] = image_path
                    result['image_name'] = os.path.basename(image_path)
                    result['high_confidence'] = result['confidence'] >= min_confidence
                    results.append(result)
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"   Đã xử lý {i + 1}/{len(image_files)} ảnh")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Lỗi với ảnh {image_path}: {e}")
                    failed_images.append(image_path)
            
            # Thống kê kết quả
            total_processed = len(results)
            high_confidence_count = sum(1 for r in results if r['high_confidence'])
            class_counts = {}
            for class_name in self.class_names:
                class_counts[class_name] = sum(1 for r in results if r['predicted_class'] == class_name)
            
            logger.info(f"📊 Kết quả batch prediction:")
            logger.info(f"   Tổng ảnh xử lý: {total_processed}")
            logger.info(f"   Ảnh failed: {len(failed_images)}")
            logger.info(f"   Ảnh high confidence (>={min_confidence}): {high_confidence_count}")
            logger.info(f"   Phân bố class:")
            for class_name, count in class_counts.items():
                logger.info(f"     {class_name}: {count} ảnh")
            
            # Lưu CSV nếu được yêu cầu
            if output_csv and results:
                self._save_batch_results_to_csv(results, output_csv)
            
            return {
                'results': results,
                'failed_images': failed_images,
                'statistics': {
                    'total_processed': total_processed,
                    'total_failed': len(failed_images),
                    'high_confidence_count': high_confidence_count,
                    'class_distribution': class_counts
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi dự đoán batch: {e}")
            raise
    
    def _save_batch_results_to_csv(self, results, output_csv):
        """Lưu kết quả batch vào CSV"""
        try:
            import pandas as pd
            
            # Chuẩn bị data cho CSV
            csv_data = []
            for result in results:
                row = {
                    'image_path': result['image_path'],
                    'image_name': result['image_name'],
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'high_confidence': result['high_confidence']
                }
                
                # Thêm confidence cho từng class
                for class_name, conf in result['all_predictions'].items():
                    row[f'confidence_{class_name}'] = conf
                
                csv_data.append(row)
            
            # Tạo DataFrame và lưu
            df = pd.DataFrame(csv_data)
            df.to_csv(output_csv, index=False, encoding='utf-8')
            
            logger.info(f"💾 Batch results saved to: {output_csv}")
            
        except Exception as e:
            logger.warning(f"⚠️  Không thể lưu CSV: {e}")
    
    def load_model(self, model_path):
        """Load model đã training"""
        try:
            model_path = self._normalize_path(model_path)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
            
            logger.info(f"📂 Loading model từ: {model_path}")
            
            # Load model
            self.model = keras.models.load_model(model_path)
            
            # Load training info nếu có
            info_path = model_path.replace('.keras', '_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.training_config = json.load(f)
                logger.info("📄 Đã load training config")
            
            logger.info("✅ Model loaded thành công!")
            
            # In thông tin model
            logger.info(f"📊 Model info:")
            logger.info(f"   Total parameters: {self.model.count_params():,}")
            logger.info(f"   Input shape: {self.model.input_shape}")
            logger.info(f"   Output shape: {self.model.output_shape}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi load model: {e}")
            raise
    
    def save_model(self, save_path):
        """Lưu model"""
        try:
            if self.model is None:
                raise ValueError("Không có model để lưu!")
            
            save_path = self._normalize_path(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Lưu model
            self.model.save(save_path)
            logger.info(f"💾 Model đã lưu tại: {save_path}")
            
            # Lưu training config nếu có
            if self.training_config:
                info_path = save_path.replace('.keras', '_info.json')
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(self.training_config, f, indent=2, ensure_ascii=False)
                logger.info(f"📄 Training config đã lưu tại: {info_path}")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu model: {e}")
            raise


# Ví dụ sử dụng
def main():
    """Ví dụ sử dụng Enhanced Phone Detection CNN"""
    try:
        # Khởi tạo model
        print("🚀 Khởi tạo Enhanced Phone Detection CNN...")
        phone_detector = EnhancedPhoneDetectionCNN(
            img_height=224,
            img_width=224,
            use_pretrained=True  # Sử dụng Transfer Learning
        )
        
        # Đường dẫn dữ liệu (thay đổi theo đường dẫn thực tế của bạn)
        train_data_dir = "data/train"  # Thư mục chứa 2 folder: no_phone và using_phone
        test_data_dir = "data/test"    # Tương tự cho test
        
        # Tạo model
        print("\n🏗️ Xây dựng model...")
        model = phone_detector.build_model(
            num_classes=2,
            dropout_rate=0.3
        )
        
        # Tạo data generators
        print("\n📊 Tạo data generators...")
        train_gen, val_gen = phone_detector.create_data_generators(
            train_dir=train_data_dir,
            val_dir=None,  # Sẽ tự động split từ train
            batch_size=32,
            validation_split=0.2
        )
        
        # Training
        print("\n🚀 Bắt đầu training...")
        history = phone_detector.train(
            train_generator=train_gen,
            val_generator=val_gen,
            epochs=100,  # tăng số epoch
            learning_rate=0.0005,  # giảm learning rate
            save_path='models/enhanced_phone_detection_model.keras',
            use_class_weights=True,
            patience=20,  # tăng patience cho early stopping
            fine_tune_epochs=20  # tăng fine-tune epochs
        )
        
        # Đánh giá model
        if os.path.exists(test_data_dir):
            print("\n🧪 Đánh giá model trên test set...")
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_gen = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                classes=phone_detector.class_names,
                shuffle=False
            )
            
            evaluation_results = phone_detector.evaluate_model(
                test_gen,
                save_results=True,
                save_path='models/enhanced_phone_detection_model.keras'
            )
            
            print(f"\n📊 Final Test Results:")
            print(f"   Accuracy: {evaluation_results['accuracy']:.4f}")
            print(f"   Precision: {evaluation_results['precision']:.4f}")
            print(f"   Recall: {evaluation_results['recall']:.4f}")
            print(f"   F1 Score: {evaluation_results['f1_score']:.4f}")
        
        # Ví dụ predict single image
        print("\n🎯 Ví dụ predict single image...")
        sample_image = "sample_images/test_image.jpg"  # Thay đổi đường dẫn
        if os.path.exists(sample_image):
            result = phone_detector.predict_single_image(
                sample_image,
                show_result=True
            )
            print(f"Kết quả: {result['predicted_class']} (confidence: {result['confidence']:.4f})")
        
        # Ví dụ batch prediction
        print("\n📁 Ví dụ batch prediction...")
        batch_folder = "sample_images/"  # Thư mục chứa nhiều ảnh test
        if os.path.exists(batch_folder):
            batch_results = phone_detector.predict_batch_images(
                batch_folder,
                output_csv="results/batch_predictions.csv",
                min_confidence=0.7
            )
            
            print(f"Đã xử lý {batch_results['statistics']['total_processed']} ảnh")
            print(f"High confidence predictions: {batch_results['statistics']['high_confidence_count']}")
        
        print("\n✅ Hoàn thành!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Lỗi file/folder: {e}")
        print("💡 Hướng dẫn:")
        print("   1. Tạo thư mục data/train với 2 subfolder: no_phone và using_phone")
        print("   2. Thêm ảnh training vào các subfolder tương ứng")
        print("   3. (Tùy chọn) Tạo data/test với cấu trúc tương tự cho evaluation")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()