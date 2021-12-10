PRESET = {
    "global_configs": {
        "gln_globals": {
            "output_folder": "gln_run",
            "n_epochs": 20,
            "lr": 0.001,
            "sample_interval": 300,
            "checkpoint_interval": 300,
            "n_saved_models": 3,
            "dataloader_workers": 0,
            "device": "cpu",
            "get_acts": True,
            "memory_dataset": False,
        },
    },
    "input_configs": {
        "gln_input": {
            "input_info": {
                "input_source": "MUST_FILL",
                "input_name": "genotype",
                "input_type": "omics",
            },
            "input_type_info": {
                "na_augment_perc": 0.4,
                "na_augment_prob": 1.0,
                "snp_file": "MUST_FILL",
            },
            "model_config": {
                "model_type": "genome-local-net",
                "model_init_config": {
                    "rb_do": 0.5,
                    "channel_exp_base": 2,
                    "layers": [2],
                    "kernel_width": 8,
                },
            },
        }
    },
    "predictor_configs": {
        "gln_predictor": {
            "model_type": "default",
            "model_config": {
                "layers": [2],
                "fc_do": 0.5,
                "rb_do": 0.5,
                "fc_task_dim": 64,
            },
        }
    },
    "target_configs": {
        "gln_targets": {"label_file": "MUST_FILL", "target_cat_columns": "MUST_FILL"}
    },
}
