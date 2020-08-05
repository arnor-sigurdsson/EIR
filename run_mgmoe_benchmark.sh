###  MGMoE  ### 
# for trait in gout diabetes_1 diabetes_2 hypothyroidism hypertension asthma glaucoma atrial_fibrillation heart_attack; do
for trait in gout diabetes_1 diabetes_2 hypothyroidism asthma glaucoma atrial_fibrillation heart_attack; do
       python human_origins_supervised/train.py --run_name EXPERIMENTS/MLP-MGMoE_BASELINES_LR-1e-04_L1-1e-02_WD-1e-02_LARGE-VALID/0_"$trait"_MLP_MISSING_LR-5e-5_L1-1e-3_COSINE --l1 1e-02 --lr 1e-04 --gpu_num 1 --model_type mlp-mgmoe --label_file data/pre_computed_many_diseases.csv --valid_size 50000 --sample_interval 400 --target_cat_columns "$trait" --weighted_sampling_column all --lr_schedule cosine --optimizer adamw --data_source data/UKBB_IMPUTED_NO-MISSING/processed/parsed_files/full_indiv/full_snps/train_paths_no_missing.txt --n_epochs 50 --warmup auto --fc_repr_dim 1 --fc_task_dim 32 --resblocks 2 2 --wd 1e-02 --rb_do 0.5 --fc_do 0.5 --mg_num_experts 1 --split_mlp_num_splits 1024
done
