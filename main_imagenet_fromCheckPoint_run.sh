# Default arguments
seed=1005
arch="resnet18"
batch_size=64
workers=8
data_path="data/ImageNet"

# Quantization parameters
n_bits_w=8
channel_wise=True
n_bits_a=8

# Weight calibration parameters
num_samples=1024
iters_w=20000
weight=0.01

b_start=20
b_end=2
warmup=0.2

# Activation calibration parameters
lr=4e-5

init_wmode="mse"
init_amode="mse"

prob=0.5
input_prob=0.5
lamb_r=0.1
T=4.0
bn_lr=1e-3
lamb_c=0.02

filename="default" # Filename to save the model / 24.05.29 @jiho264
# Logging arguments to a file
log_file="logs/W${n_bits_w}A${n_bits_a}_calib${num_samples}_batch${batch_size}_iterW${iters_w}/${arch}/${filename}_fromPTH.log"
# Clear the log file
> $log_file
echo "START : $(date +%Y)-$(date +%m)-$(date +%d) $(date +%H):$(date +%M):$(date +%S)" >> $log_file
echo "" >> $log_file

echo "General parameters for data and model" >> $log_file
echo "- seed = $seed (default = 1005)" >> $log_file
echo "- arch = $arch" >> $log_file
echo "- batch_size = $batch_size (default = 64)" >> $log_file
echo "- workers = $workers (default = 4)" >> $log_file
echo "- data_path = $data_path" >> $log_file
echo "" >> $log_file
# Quantization parameters
echo "Quantization parameters" >> $log_file
echo "- n_bits_w = $n_bits_w (default = 4)" >> $log_file
echo "- channel_wise = $channel_wise (default = True)" >> $log_file
echo "- n_bits_a = $n_bits_a (default = 4)" >> $log_file
echo "- disable_8bit_head_stem = not use (action = 'store_true')" >> $log_file
echo "" >> $log_file
# Weight calibration parameters
echo "Weight calibration parameters" >> $log_file
echo "- num_samples = $num_samples (default = 1024)" >> $log_file
echo "- iters_w = $iters_w (default = 20000)" >> $log_file
echo "- weight = $weight (default = 0.01)" >> $log_file
echo "- keep_cpu = not use (action = 'store_true')" >> $log_file

echo "- b_start = $b_start (default = 20)" >> $log_file
echo "- b_end = $b_end (default = 2)" >> $log_file
echo "- warmup = $warmup (default = 0.2)" >> $log_file
echo "" >> $log_file
# Activation calibration parameters
echo "Activation calibration parameters" >> $log_file
echo "- lr = $lr (default = 4e-5)" >> $log_file

echo "- init_wmode = $init_wmode (default = 'mse', choices = ['minmax', 'mse', 'minmax_scale'])" >> $log_file
echo "- init_amode = $init_amode (default = 'mse', choices = ['minmax', 'mse', 'minmax_scale'])" >> $log_file

echo "- prob = $prob (default = 0.5)" >> $log_file
echo "- input_prob = $input_prob (default = 0.5)" >> $log_file
echo "- lamb_r = $lamb_r (default = 0.1)" >> $log_file
echo "- T = $T (default = 4.0)" >> $log_file
echo "- bn_lr = $bn_lr (default = 1e-3)" >> $log_file
echo "- lamb_c = $lamb_c (default = 0.02)" >> $log_file
echo "" >> $log_file
echo "-------------------------------------------------------------------------------------------" >> $log_file
echo "" >> $log_file

# Execute the main script
python main_imagenet_fromCheckPoint.py \
    --seed $seed \
    --arch $arch \
    --batch_size $batch_size \
    --workers $workers \
    --data_path $data_path \
    --n_bits_w $n_bits_w \
    --channel_wise $channel_wise \
    --n_bits_a $n_bits_a \
    --disable_8bit_head_stem $disable_8bit_head_stem \
    --num_samples $num_samples \
    --iters_w $iters_w \
    --weight $weight \
    --keep_cpu $keep_cpu \
    --b_start $b_start \
    --b_end $b_end \
    --warmup $warmup \
    --lr $lr \
    --init_wmode $init_wmode \
    --init_amode $init_amode \
    --prob $prob \
    --input_prob $input_prob \
    --lamb_r $lamb_r \
    --T $T \
    --bn_lr $bn_lr \
    --lamb_c $lamb_c \
    --filename $filename \
    | tee -a $log_file

echo "END : $(date +%Y)-$(date +%m)-$(date +%d) $(date +%H):$(date +%M):$(date +%S)" >> $log_file