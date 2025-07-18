DATA_DIR="/data1035/yuxuanlong/codebase/rf100-vl/rf100-vl"
CONFIG_TEMPLATE="/data/liulongfei/DEIM/configs/deim_rtdetrv2/rf100vl_ft.yml"
SAVE_ROOT="/data/liulongfei/DEIM/outputs/rf100vl"
TEMP_CONFIG="/data/liulongfei/DEIM/configs/deim_rtdetrv2/temp_config.yml"


declare -A DATASET_CLASSES=(
  ["-grccs"]=7
  ["13-lkc01"]=1
  ["2024-frc"]=11
  ["actions"]=6
  ["activity-diagrams"]=12
  ["aerial-airport"]=1
  ["aerial-cows"]=1
  ["aerial-pool"]=5
  ["aerial-sheep"]=1
  ["aircraft-turnaround-dataset"]=5
  ["all-elements"]=10
  ["apoce-aerial-photographs-for-object-detection-of-construction-equipment"]=7
  ["aquarium-combined"]=7
  ["asphaltdistressdetection"]=6
  ["ball"]=18
  ["bees"]=1
  ["bibdetection"]=1
  ["buoy-onboarding"]=7
  ["cable-damage"]=2
  ["canalstenosis"]=5
  ["car-logo-detection"]=2
  ["circuit-voltages"]=3
  ["clashroyalechardetector"]=34
  ["cod-mw-warzone"]=2
  ["conveyor-t-shirts"]=1
  ["countingpills"]=1
  ["crystal-clean-brain-tumors-mri-dataset"]=4
  ["dataconvert"]=1
  ["deepfruits"]=1
  ["deeppcb"]=6
  ["defect-detection"]=4
  ["dentalai"]=4
  ["electric-pylon-detection-in-rsi"]=1
  ["everdaynew"]=5
  ["exploratorium-daphnia"]=13
  ["flir-camera-objects"]=4
  ["floating-waste"]=5
  ["football-player-detection"]=2
  ["fruitjes"]=7
  ["grapes-5"]=2
  ["grass-weeds"]=1
  ["gwhd2021"]=1
  ["halo-infinite-angel-videogame"]=4
  ["human-detection-in-floods"]=2
  ["inbreast"]=1
  ["infraredimageofpowerequipment"]=9
  ["into-the-vale"]=6
  ["invoice-processing"]=8
  ["ism-band-packet-detection"]=3
  ["jellyfish"]=1
  ["l10ul502"]=8
  ["label-printing-defect-version-2"]=2
  ["lacrosse-object-detection"]=4
  ["liver-disease"]=4
  ["macro-segmentation"]=16
  ["mahjong"]=34
  ["marine-sharks"]=4
  ["needle-base-tip-min-max"]=5
  ["new-defects-in-wood"]=5
  ["nih-xray"]=8
  ["orgharvest"]=2
  ["orionproducts"]=8
  ["paper-parts"]=19
  ["peixos-fish"]=2
  ["penguin-finder-seg"]=1
  ["pig-detection"]=4
  ["pill"]=5
  ["recode-waste"]=6
  ["roboflow-trained-dataset"]=4
  ["screwdetectclassification"]=5
  ["sea-cucumbers-new-tiles"]=1
  ["signatures"]=1
  ["smd-components"]=11
  ["soda-bottles"]=3
  ["speech-bubbles-detection"]=6
  ["spinefrxnormalvindr"]=3
  ["sssod"]=2
  ["stomata-cells"]=2
  ["taco-trash-annotations-in-context"]=22
  ["the-dreidel-project"]=6
  ["thermal-cheetah"]=1
  ["tomatoes-2"]=2
  ["trail-camera"]=2
  ["train"]=2
  ["truck-movement"]=5
  ["tube"]=5
  ["uavdet-small"]=7
  ["ufba-425"]=32
  ["underwater-objects"]=5
  ["urine-analysis1"]=5
  ["varroa-mites-detection--test-set"]=1
  ["water-meter"]=10
  ["wb-prova"]=3
  ["weeds4"]=5
  ["wheel-defect-detection"]=4
  ["wildfire-smoke"]=1
  ["wine-labels"]=11
  ["x-ray-id"]=6
  ["xray"]=1
  ["zebrasatasturias"]=1
)


for DATA_PATH in "$DATA_DIR"/*; do
    DATA_NAME=$(basename "$DATA_PATH")        
    NAME_NO_EXT="${DATA_NAME%.*}"            
    SAVE_PATH="$SAVE_ROOT/$DATA_NAME"

    TRAIN_IMG_FOLDER="$DATA_PATH/train"
    TRAIN_ANN_FILE="$DATA_PATH/train/_annotations.coco.json"
    VAL_IMG_FOLDER="$DATA_PATH/valid"
    VAL_ANN_FILE="$DATA_PATH/valid/_annotations.coco.json"

    NUM_CLASSES=${DATASET_CLASSES[$NAME_NO_EXT]}
    
    echo "$SAVE_PATH"

    LOG="rf100vl/$DATA_NAME.log"

    sed -e "s|img_folder: .*train2017/|img_folder: ${TRAIN_IMG_FOLDER}/|" \
        -e "s|ann_file: .*instances_train2017.json|ann_file: ${TRAIN_ANN_FILE}|" \
        -e "s|img_folder: .*val2017/|img_folder: ${VAL_IMG_FOLDER}/|" \
        -e "s|ann_file: .*instances_val2017.json|ann_file: ${VAL_ANN_FILE}|" \
        -e "s|output_dir:.*|output_dir: ${SAVE_PATH}|" \
        -e "s|num_classes:.*|num_classes: ${NUM_CLASSES}|" \
        "$CONFIG_TEMPLATE" > "$TEMP_CONFIG"

    CUDA_VISIBLE_DEVICES=9 torchrun --master_port=7777 --nproc_per_node=1 train.py \
     -c "$TEMP_CONFIG" --use-amp --seed=0 \
     -t /data1032/liulongfei/OV-DEIM-COCO/checkpoints/ovdeim.pth 2>&1 | tee "$LOG"
    find /data/liulongfei/DEIM/outputs/rf100vl -name *.pth -type f -delete

done
