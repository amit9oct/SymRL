# Run the main.py script on the best model found during training
import os

# Create a subproces to run the test.py script
best_model_path = ".models/best_train_model"
args = f"--no_train --no_test --gui --load --folder {best_model_path}"
file_dir_path = os.path.dirname(os.path.realpath(__file__))
py_file = os.path.join(file_dir_path, "main.py")
cmd = f"python {py_file} {args}"
print(f"Running the following command: {cmd}")
os.system(cmd)