alpha=0.001
train_cnt=100
test_cnt=25
train_max_terms=10
test_max_terms=12
float_prob=0.12
frac_prob=0.2
num_episodes=150000
maximum_step_limit=10
nohup python src/symrl/main.py --exp_prefix var_const_cnt --alpha $alpha --feat_ex var_const_count --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob $float_prob --frac_prob $frac_prob --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &
nohup python src/symrl/main.py --exp_prefix term_var_const_cnt --alpha $alpha --feat_ex term_var_const_count --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob $float_prob --frac_prob $frac_prob --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &
nohup python src/symrl/main.py --exp_prefix op_var_cnt --alpha $alpha --feat_ex op_var_count --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob $float_prob --frac_prob $frac_prob --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &
nohup python src/symrl/main.py --exp_prefix op_cnt --alpha $alpha --feat_ex op_count --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob $float_prob --frac_prob $frac_prob --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &
nohup python src/symrl/main.py --exp_prefix term_var_const_gtr_cnt --alpha $alpha --feat_ex simpl_term_var_const_count --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob $float_prob --frac_prob $frac_prob --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &
nohup python src/symrl/main.py --exp_prefix int_term_var_const_cnt --alpha $alpha --feat_ex term_var_const_count --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob 0.0 --frac_prob 0.2 --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &
nohup python src/symrl/main.py --exp_prefix int_term_var_const_gtr_cnt --alpha $alpha --feat_ex simpl_term_var_const_count --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob 0.0 --frac_prob 0.2 --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &
nohup python src/symrl/main.py --exp_prefix human --no_test --no_train --do_human_test --do_human_train --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob $float_prob --frac_prob $frac_prob --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &
nohup python src/symrl/main.py --exp_prefix random --no_test --no_train --do_random_test --do_random_train --train_cnt $train_cnt --test_cnt $test_cnt --train_max_terms $train_max_terms --test_max_terms $test_max_terms --float_prob $float_prob --frac_prob $frac_prob --num_episodes $num_episodes --maximum_step_limit $maximum_step_limit &