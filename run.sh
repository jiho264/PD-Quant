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
    --arch $arch \
    --n_bits_w $w_bits \
    --n_bits_a $a_bits \
    --weight $weight \
    --T $T \
    --lamb_c $lamb_c \
    --workers 8 \
    | tee "logs/resnet18_W"$w_bits"A"$a_bits".log"



# python run_script.py resnet18