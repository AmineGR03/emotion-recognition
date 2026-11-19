# Entrainement du modele leger (TinyEmotionCNN) sur CPU
python -u src/train.py --data-dir data/fer2013 --arch tiny --epochs 60 --batch-size 64 --num-workers 4 --weight-power 0.5 --run-name tiny 2>&1 | Tee-Object -FilePath outputs_train_tiny.log
