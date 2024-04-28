# Run the main.py script on the best model found during training
import os

# Create a subproces to run the test.py script
best_model_path = ".models/best_train_model"
args = f"--no_train --load --folder {best_model_path}"
file_dir_path = os.path.dirname(os.path.realpath(__file__))
py_file = os.path.join(file_dir_path, "main.py")
cmd = f"python {py_file} {args}"
print(f"Running the following command: {cmd}")
results_folder = "src/.logs/_test__term_var_const_gtr_cnt_lin__td__gr"
print(f"Lookup the folder {results_folder} for the results of the test run.")
os.system(cmd)

