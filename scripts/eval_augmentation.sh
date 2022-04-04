echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                   Training Start (Aug 100%)                * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Make training label.
./convert_labels.sh kitti
# Train VGG16Net.
./train.sh

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                  Training Complete (Aug 100%)              * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Make new directory.
mkdir ../../log/data_augmentation/new_100
# Plot loss and move.
python plot_loss.py --file loss_*
mv loss_po* ../../log/data_augmentation/new_100/
# Copy trained weights.
cp ../model/out/VGG16_faster_rcnn_iter_60000.caffemodel ../../log/data_augmentation/new_100/
cp ../model/out/VGG16_faster_rcnn_iter_70000.caffemodel ../../log/data_augmentation/new_100/

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                    70k evaluation (Aug 100%)               * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Convert model - 70000.
./convert_model.sh 70000
# Do evaluation.
./evaluate.sh
# Move result files.
mv Result_precision_recall.jpg ../../log/data_augmentation/new_100/Result_precision_recall_70k.jpg
mkdir ../../log/data_augmentation/new_100/70k
mv Result_rad* ../../log/data_augmentation/new_100/70k/

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                    60k evaluation (Aug 100%)               * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Convert model - 60000.
./convert_model.sh 60000
# Do evaluation.
./evaluate.sh
# Move result files.
mv Result_precision_recall.jpg ../../log/data_augmentation/new_100/Result_precision_recall_60k.jpg
mkdir ../../log/data_augmentation/new_100/60k
mv Result_rad* ../../log/data_augmentation/new_100/60k/

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                        Delete labels                       * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Delete augmented files - delete after 2800
rm /media/dataset/kitti/2d/training_label/0028??_aug*
rm /media/dataset/kitti/2d/training_label/0029??_aug*
rm /media/dataset/kitti/2d/training_label/003???_aug*
rm /media/dataset/kitti/2d/training_label/004???_aug*
rm /media/dataset/kitti/2d/training_label/005???_aug*

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                   Training Start (Aug 50%)                 * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Make training label.
./convert_labels.sh kitti
# Train VGG16Net.
./train.sh

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                  Training Complete (Aug 50%)               * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Make new directory.
mkdir ../../log/data_augmentation/new_50
# Plot loss and move.
python plot_loss.py --file loss_*
mv loss_po* ../../log/data_augmentation/new_50/
# Copy trained weights.
cp ../model/out/VGG16_faster_rcnn_iter_60000.caffemodel ../../log/data_augmentation/new_50/
cp ../model/out/VGG16_faster_rcnn_iter_70000.caffemodel ../../log/data_augmentation/new_50/

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                    70k evaluation (Aug 50%)                * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Convert model - 70000.
./convert_model.sh 70000
# Do evaluation.
./evaluate.sh
# Move result files.
mv Result_precision_recall.jpg ../../log/data_augmentation/new_50/Result_precision_recall_70k.jpg
mkdir ../../log/data_augmentation/new_50/70k
mv Result_rad* ../../log/data_augmentation/new_50/70k/

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                    60k evaluation (Aug 50%)                * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Convert model - 60000.
./convert_model.sh 60000
# Do evaluation.
./evaluate.sh
# Move result files.
mv Result_precision_recall.jpg ../../log/data_augmentation/new_50/Result_precision_recall_60k.jpg
mkdir ../../log/data_augmentation/new_50/60k
mv Result_rad* ../../log/data_augmentation/new_50/60k/

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                        Delete labels                       * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Delete augmented files - delete after 1401
rm /media/dataset/kitti/2d/training_label/001402_aug*
rm /media/dataset/kitti/2d/training_label/001403_aug*
rm /media/dataset/kitti/2d/training_label/001404_aug*
rm /media/dataset/kitti/2d/training_label/001405_aug*
rm /media/dataset/kitti/2d/training_label/001406_aug*
rm /media/dataset/kitti/2d/training_label/001407_aug*
rm /media/dataset/kitti/2d/training_label/001408_aug*
rm /media/dataset/kitti/2d/training_label/001409_aug*
rm /media/dataset/kitti/2d/training_label/00141?_aug*
rm /media/dataset/kitti/2d/training_label/00142?_aug*
rm /media/dataset/kitti/2d/training_label/00143?_aug*
rm /media/dataset/kitti/2d/training_label/00144?_aug*
rm /media/dataset/kitti/2d/training_label/00145?_aug*
rm /media/dataset/kitti/2d/training_label/00146?_aug*
rm /media/dataset/kitti/2d/training_label/00147?_aug*
rm /media/dataset/kitti/2d/training_label/00148?_aug*
rm /media/dataset/kitti/2d/training_label/00149?_aug*
rm /media/dataset/kitti/2d/training_label/0015??_aug*
rm /media/dataset/kitti/2d/training_label/0016??_aug*
rm /media/dataset/kitti/2d/training_label/0017??_aug*
rm /media/dataset/kitti/2d/training_label/0018??_aug*
rm /media/dataset/kitti/2d/training_label/0019??_aug*
rm /media/dataset/kitti/2d/training_label/002???_aug*

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                   Training Start (Aug 25%)                 * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Make training label.
./convert_labels.sh kitti
# Train VGG16Net.
./train.sh

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                  Training Complete (Aug 25%)               * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Make new directory.
mkdir ../../log/data_augmentation/new_25
# Plot loss and move.
python plot_loss.py --file loss_*
mv loss_po* ../../log/data_augmentation/new_25/
# Copy trained weights.
cp ../model/out/VGG16_faster_rcnn_iter_60000.caffemodel ../../log/data_augmentation/new_25/
cp ../model/out/VGG16_faster_rcnn_iter_70000.caffemodel ../../log/data_augmentation/new_25/

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                    70k evaluation (Aug 25%)                * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Convert model - 70000.
./convert_model.sh 70000
# Do evaluation.
./evaluate.sh
# Move result files.
mv Result_precision_recall.jpg ../../log/data_augmentation/new_25/Result_precision_recall_70k.jpg
mkdir ../../log/data_augmentation/new_25/70k
mv Result_rad* ../../log/data_augmentation/new_25/70k/

echo -e "\e[32m ************************************************************** \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m *                    60k evaluation (Aug 25%)                * \e[39m"
echo -e "\e[32m *                                                            * \e[39m"
echo -e "\e[32m ************************************************************** \e[39m"

# Convert model - 60000.
./convert_model.sh 60000
# Do evaluation.
./evaluate.sh
# Move result files.
mv Result_precision_recall.jpg ../../log/data_augmentation/new_25/Result_precision_recall_60k.jpg
mkdir ../../log/data_augmentation/new_25/60k
mv Result_rad* ../../log/data_augmentation/new_25/60k/


