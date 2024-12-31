learning_rates=(0.03)
dropouts=(0.1)
batch_sizes=(64 128)
n_layers=(1)

available_gpus=(0 1 2 3 4 5 6 7) # 92
# available_gpus=(4 5 6 7)
# available_gpus=(0 1 2 3)
# available_gpus=( 3 )
declare -A pid_to_gpu

for lr in "${learning_rates[@]}"; do
    for dp in "${dropouts[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            for n_layer in "${n_layers[@]}"; do
            #     for w_div in "${w_divs[@]}"; do
                    while [[ ${#available_gpus[@]} -eq 0 ]]; do
                        wait -n
                        running_pid=$(jobs -p)

                        for key in "${!pid_to_gpu[@]}"; do
                            if [[ ${running_pid[@]/${key}/} == ${running_pid[@]} ]]; then
                                gpu=${pid_to_gpu[$key]}
                                available_gpus+=("$gpu")
                                unset pid_to_gpu[$key]
                                break
                            fi
                        done
                    done

                    next_gpu="${available_gpus[0]}"
                    available_gpus=("${available_gpus[@]:1}")
                    command="python train.py --learning_rate=$lr --dropout=$dp --batch_size=$bs --visible_gpu=$next_gpu"

                    echo "Executing: $command"
                    python train.py --n_layer=$n_layer --learning_rate=$lr --dropout=$dp --batch_size=$bs --visible_gpu=$next_gpu &
                    current_pid=$!
                    pid_to_gpu["$current_pid"]="$next_gpu"
                    echo "pid_to_gpu: ${!pid_to_gpu[@]}"
                    echo "pid_to_gpu: ${pid_to_gpu[@]}"
                    sleep 10
                    
            #     done
            done
        done
    done
done