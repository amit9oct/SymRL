# Run the main.py script on the best model found during training
# We need to run 
# nohup python src/symrl/test.py --exp_prefix term_var_const_gtr_cnt --alpha $alpha --feat_ex simpl_term_var_const_count --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob $float_prob --frac_prob $frac_prob --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &
import os

alpha=0.001
train_cnt=100
test_cnt=25
train_max_terms=10
test_max_terms=12
float_prob=0.12
frac_prob=0.2
num_episodes=150000
maximum_step_limit=10
# Create a subproces to run the main.py script
args = f"--exp_prefix term_var_const_gtr_cnt --alpha {alpha} --feat_ex simpl_term_var_const_count --train_cnt {train_cnt} --test_cnt {test_cnt} --train_max_terms {train_max_terms} --test_max_terms {test_max_terms} --float_prob {float_prob} --frac_prob {frac_prob} --num_episodes {num_episodes} --maximum_step_limit {maximum_step_limit}"
file_dir_path = os.path.dirname(os.path.realpath(__file__))
py_file = os.path.join(file_dir_path, "main.py")
cmd = f"python {py_file} {args}"
print(f"Running the following command: {cmd}")
results_folder = "src/.logs/term_var_const_gtr_cnt_train__simpl_term_var_const_count_lin__td__eps"
print(f"Lookup the folder {results_folder} for the results of the test run.")
os.system(cmd)

