for run_id in 3 4 #2 3 4 5
do
    nohup python -u train_grammarly.py $run_id > "run_id_tree_only_${run_id}.out" &
done
