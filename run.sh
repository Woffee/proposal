log_file=2020-11-10-n100.log

for nodes_num in 120
do
  for obs_num in 80 90 100 110 120
  do
    python main_without_iteration_vali.py --nodes_num $nodes_num --obs_num $obs_num --log $log_file
  done
done