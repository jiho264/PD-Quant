#    w_bits = [2, 4, 2, 4]
#    a_bits = [2, 2, 4, 4]

arch=resnet18
w_bits=4
a_bits=4
weight=0.01
T=4.0
lamb_c=0.02

python main_imagenet.py \
    --data_path data/ImageNet \
    --batch_size 16 \
    --num_samples 256 \
    --arch $arch \
    --n_bits_w $w_bits \
    --n_bits_a $a_bits \
    --weight $weight \
    --T $T \
    --lamb_c $lamb_c \
    --workers 8 \
    | tee "logs/resnet18_W"$w_bits"A"$a_bits"batch16_calib256.log"



# python run_script.py resnet18