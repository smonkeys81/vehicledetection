# This script test single image via vehicle detection module based on faster RCNN.

# arguments:
# 1. Configuration file.
# 2. Network model name.
# 3. Weight file.
# 4. Path to test image.

../tools/test_single_image  ../config/config.json \
                            ZF \
                            ../model/out/ZF_faster_rcnn_final.caffemodel \
                            ../test/sample_img/000008.png
