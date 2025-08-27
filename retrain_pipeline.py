import tensorflow as tf
import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import shutil
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.model_base_path = self.config.get('model_base_path', './models/mnist_model')
        self.accuracy_threshold = self.config.get('accuracy_threshold', 0.97)
        self.backup_path = self.config.get('backup_path', './models/backups')
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def send_notification(self, subject, message, is_success=True):
        """Send email notification using environment variables for security"""
        try:
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            sender_email = os.getenv('SENDER_EMAIL')
            sender_password = os.getenv('SENDER_PASSWORD')
            recipient_email = os.getenv('RECIPIENT_EMAIL')
            
            if not all([sender_email, sender_password, recipient_email]):
                logger.warning("Email credentials not configured. Skipping notification.")
                return
            
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = recipient_email
            
            # Add timestamp and status
            full_message = f"""
            Timestamp: {datetime.datetime.now().isoformat()}
            Status: {'SUCCESS' if is_success else 'FAILED'}
            
            {message}
            
            Model Base Path: {self.model_base_path}
            Accuracy Threshold: {self.accuracy_threshold * 100:.1f}%
            """
            
            msg.attach(MIMEText(full_message, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                
            logger.info("Notification email sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")
    
    def backup_current_model(self, current_version):
        """Create backup of current model before promoting new version"""
        try:
            if current_version == 0:
                return True
                
            source_path = f"{self.model_base_path}/{current_version}"
            backup_dir = Path(self.backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"model_v{current_version}_{timestamp}"
            
            shutil.copytree(source_path, backup_path)
            logger.info(f"Model v{current_version} backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup model: {str(e)}")
            return False
    
    def get_current_version(self):
        """Get the current highest model version"""
        try:
            if not os.path.exists(self.model_base_path):
                os.makedirs(self.model_base_path, exist_ok=True)
                return 0
                
            versions = [int(v) for v in os.listdir(self.model_base_path) 
                       if v.isdigit() and os.path.isdir(f"{self.model_base_path}/{v}")]
            return max(versions) if versions else 0
        except Exception as e:
            logger.error(f"Error getting current version: {str(e)}")
            return 0
    
    def save_model_metadata(self, version, accuracy, training_time, promoted=True):
        """Save model metadata for tracking"""
        metadata = {
            'version': version,
            'accuracy': float(accuracy),
            'training_time': training_time.isoformat(),
            'promoted': promoted,
            'threshold': self.accuracy_threshold
        }
        
        metadata_path = f"{self.model_base_path}/metadata_v{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved for version {version}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST data"""
        logger.info("Loading MNIST dataset...")
        
        try:
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            
            # Normalize pixel values
            x_train, x_test = x_train / 255.0, x_test / 255.0
            
            # Add channel dimension
            x_train = x_train[..., tf.newaxis]
            x_test = x_test[..., tf.newaxis]
            
            logger.info(f"Data loaded: Train={x_train.shape}, Test={x_test.shape}")
            return (x_train, y_train), (x_test, y_test)
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def build_model(self):
        """Build the CNN model architecture"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, train_data, test_data, epochs=10):
        """Train the model with early stopping"""
        (x_train, y_train) = train_data
        (x_test, y_test) = test_data
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, test_data):
        """Evaluate model and return accuracy"""
        (x_test, y_test) = test_data
        
        logger.info("Evaluating model...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        logger.info(f"Test accuracy: {test_acc * 100:.2f}%")
        return test_acc
    
    def promote_model(self, model, new_version):
        """Save and promote the new model version"""
        export_path = f'{self.model_base_path}/{new_version}'
        
        try:
            tf.saved_model.save(model, export_path)
            logger.info(f"Model v{new_version} saved to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def rollback_model(self, target_version):
        """Rollback to a previous model version"""
        try:
            target_path = f"{self.model_base_path}/{target_version}"
            if not os.path.exists(target_path):
                logger.error(f"Target version {target_version} not found")
                return False
            
            logger.info(f"Rolling back to model version {target_version}")
            # In a production environment, you might want to update symlinks or 
            # restart serving containers here
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False
    
    def run_retraining(self, epochs=10):
        """Main retraining pipeline"""
        training_start = datetime.datetime.now()
        logger.info("=== Starting Model Retraining Pipeline ===")
        
        try:
            # Get current version and prepare for new version
            current_version = self.get_current_version()
            new_version = current_version + 1
            
            logger.info(f"Current version: {current_version}, New version: {new_version}")
            
            # Load data
            train_data, test_data = self.load_and_preprocess_data()
            
            # Build and train model
            model = self.build_model()
            history = self.train_model(model, train_data, test_data, epochs)
            
            # Evaluate model
            test_accuracy = self.evaluate_model(model, test_data)
            
            # Save metadata regardless of promotion
            self.save_model_metadata(new_version, test_accuracy, training_start, 
                                   promoted=(test_accuracy >= self.accuracy_threshold))
            
            # Check accuracy gate
            if test_accuracy >= self.accuracy_threshold:
                logger.info(f"✅ Accuracy {test_accuracy * 100:.2f}% >= {self.accuracy_threshold * 100:.1f}% threshold")
                
                # Backup current model
                if self.backup_current_model(current_version):
                    # Promote new model
                    if self.promote_model(model, new_version):
                        success_msg = f"Model v{new_version} promoted successfully! Accuracy: {test_accuracy * 100:.2f}%"
                        logger.info(success_msg)
                        self.send_notification("Model Retrain SUCCESS", success_msg, is_success=True)
                        return True
                    else:
                        logger.error("Failed to promote model")
                        return False
                else:
                    logger.error("Failed to backup current model")
                    return False
            else:
                failure_msg = f"Accuracy {test_accuracy * 100:.2f}% < {self.accuracy_threshold * 100:.1f}% threshold. Keeping version {current_version}."
                logger.warning(f"❌ {failure_msg}")
                self.send_notification("Model Retrain FAILED", failure_msg, is_success=False)
                return False
                
        except Exception as e:
            error_msg = f"Retraining pipeline failed: {str(e)}"
            logger.error(error_msg)
            self.send_notification("Model Retrain ERROR", error_msg, is_success=False)
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Model Retraining Pipeline')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    
    args = parser.parse_args()
    
    retrainer = ModelRetrainer(config_path=args.config)
    success = retrainer.run_retraining(epochs=args.epochs)
    
    exit(0 if success else 1)
