python train.py --pretrained_weights weights/yolov3-tiny-rs.weights --checkpoint_interval 20 --model_def config/yolov3-tiny-rs.cfg --data_config config/rs_train_config.data --batch_size 4 --save_weights yolov3-tiny-rs
python test.py --model_def config/yolov3-tiny-rs.cfg --weights_path weights/yolov3-tiny-rs.weights --data_config config/rs_train_config.data
copy weights\yolov3-tiny-rs.weights ..\rsbot\weights
yes
copy config\yolov3-tiny-rs.cfg ..\rsbot\config
yes
copy data\RS\rs.names ..\rsbot\config
yes  