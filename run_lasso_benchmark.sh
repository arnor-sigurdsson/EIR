
# for trait in gout diabetes_1 diabetes_2 hypothyroidism hypertension asthma glaucoma atrial_fibrillation heart_attack; do
for trait in gout diabetes_1 diabetes_2 hypothyroidism asthma glaucoma atrial_fibrillation heart_attack; do
       python human_origins_supervised/train.py --run_name EXPERIMENTS/0b_LASSO_BASELINES_RARE_LARGE-VALID/0b_"$trait"_LASSO_MISSING_LINEAR-LR-5e-5_L1-1e-3_COSINE_RARE  --l1 1e-03 --lr 5e-5 --gpu_num 1 --model_type linear --label_file data/pre_computed_many_diseases.csv --valid_size 50000 --sample_interval 400 --target_cat_columns "$trait" --weighted_sampling_column all --lr_schedule cosine --optimizer adamw --data_source data/UKBB_IMPUTED_RARE/processed/parsed_files/full_indiv/full_snps/train_paths_imputed_rare.txt --n_epochs 30 --warmup auto
done
