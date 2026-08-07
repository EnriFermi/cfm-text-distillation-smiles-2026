# TinyStories: all runs and all metrics

This report is generated from `tinystories_all_measurements_long.csv`. Canonical tables keep only `canonical_for_plot=True`, `valid_value=True`, and rows without `duplicate_of`. The final catalog still covers every valid metric name for every run, including training and TensorBoard time series via count/min/max/mean/latest summaries.

- Source rows: 362,804
- Runs/models: 10/10
- Measurement types: 12
- Distinct metric names: 97
- Invalid NaN rows: 20 (all explicitly marked `valid_value=False`)

## Run inventory

| model_family | model_variant | model_status | run_id | run_complete | target_training_steps | min_training_step | max_training_step | metric_count | measurement_types |
|---|---|---|---|---|---|---|---|---|---|
| BCFM-M3 | block16_partial_step40000 | incomplete_40k_of_100k_no_final_eval | 2026-07-28_12-43-24_12345_local | False | 100000 |  |  | 4 | checkpoint_metadata \| model_inventory |
| ECLD | lr1e-3_failed_ablation | failed_degenerate_partial_progress16260_checkpoint8k | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | False | 100000 | 0 | 16259 | 22 | checkpoint_metadata \| online_quality_curve \| training_curve |
| ECLD | lr1e-4_fixed_reduction | run_completed_100k_only_step20k_weights_saved | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | True | 100000 | 0 | 100000 | 23 | checkpoint_metadata \| external_generation_eval \| online_quality_curve \| training_curve |
| AR | gpt2_artifact_corrupt | checkpoint_corrupt_not_evaluable | AR_last_flat_artifact |  | 100000 |  |  | 1 | checkpoint_metadata |
| AR | byte_trained_100k | complete_100k_legacy_byte | ar_tinystories_seed12345 | True | 100000 | 1 | 100000 | 49 | checkpoint_metadata \| external_generation_eval \| internal_checkpoint_eval \| sampler_diagnostic \| tensorboard_scalar \| validation_curve |
| BD3-LM | byte_block16_trained_100k | complete_100k_legacy_byte | bd3lm_b16_tinystories_seed12345 | True | 100000 | 1 | 100000 | 63 | checkpoint_metadata \| external_generation_eval \| internal_checkpoint_eval \| sampler_diagnostic \| tensorboard_scalar \| validation_curve |
| BD3-LM | best_canonical_100k | complete_100k_full_eval | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | True | 100000 |  |  | 27 | checkpoint_metadata \| external_generation_eval \| sequence_latency_benchmark |
| gold_reference | validation_reference | reference | gold_tinystories_n2048 | True |  |  |  | 15 | external_generation_eval \| sanity_eval_duplicate |
| MDLM | best_canonical_100k | complete_100k_full_eval | mdlm_tinystories_gpt2_h384_l6_seed12345 | True | 100000 |  |  | 27 | checkpoint_metadata \| external_generation_eval \| sequence_latency_benchmark |
| CFM-M1 | full_sequence_best_step68001 | best_checkpoint_available | tinystories_a100 | True | 100000 | 0 | 99999 | 31 | checkpoint_metadata \| external_generation_eval \| generation_benchmark \| tensorboard_scalar |

## Best recorded MAUVE point per model variant

These rows are an inventory, not a fair ranking: byte-token and GPT-2 runs belong to different comparable groups/protocols.

| model_family | model_variant | model_status | run_id | evaluation_label | sampler | point | nfe | n_samples | mauve | gen_ppl | token_entropy | seq_rep_2 | distinct_2 | js_1gram | js_2gram |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | byte_trained_100k | complete_100k_legacy_byte | ar_tinystories_seed12345 | ar_last | ar | sample_ar | 256 | 512 | 0.1295738 |  | 2.941333 | 0.5097426 | 0.005154718 | 0.6905966 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | complete_100k_legacy_byte | bd3lm_b16_tinystories_seed12345 | bd3lm_b16_last | bd3lm | sample_fh_spb16 | 16 | 512 | 0.1093303 |  | 3.591062 | 0.02176101 | 0.5772037 | 0.6931472 | 0.6931472 |
| gold_reference | validation_reference | reference | gold_tinystories_n2048 | gold_tinystories_n2048 | gold | gold |  | 2048 | 1 | 6.468816 | 4.376747 | 0.1709463 | 0.1630323 | 0 | 0 |
| BD3-LM | best_canonical_100k | complete_100k_full_eval | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | BD3LM_best_sweep | bd3lm_ancestral | sample_nfe16 | 16 | 512 | 0.9507253 | 21.5111 | 4.369975 | 0.1464844 | 0.2655331 | 0.02062944 | 0.1518967 |
| CFM-M1 | full_sequence_best_step68001 | best_checkpoint_available | tinystories_a100 | m1_tinystories_base | full_cfm | argmax_nfe64 | 64 | 512 | 0.3440323 | 89.63575 | 4.552963 | 0.09636183 | 0.3368949 | 0.04099522 | 0.2138189 |
| ECLD | lr1e-4_fixed_reduction | run_completed_100k_only_step20k_weights_saved | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | ECLD_last_step20000_argmax | full_cfm | argmax_nfe128 | 128 | 512 | 0.01040985 | 143.2454 |  |  |  |  |  |
| MDLM | best_canonical_100k | complete_100k_full_eval | mdlm_tinystories_gpt2_h384_l6_seed12345 | MDLM_best_sweep | mdlm | sample_nfe64 | 64 | 512 | 0.9445849 | 17.13951 | 4.350361 | 0.1633578 | 0.2226409 | 0.0241778 | 0.1412929 |

## External generation metrics: core (all canonical points)

| model_family | model_variant | evaluation_label | sampler | point | nfe | forwards_per_sequence | n_samples | seed | mauve | frontier_integral | gen_ppl | token_entropy | scoring_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BD3-LM | best_canonical_100k | BD3LM_best_headline_n2048 | bd3lm_ancestral | sample_nfe4 | 4 | 64 | 2048 | 0 | 0.06460061 | 0.5545368 | 72.61035 | 4.427947 | 1102.3 |
| BD3-LM | best_canonical_100k | BD3LM_best_native | bd3lm_first_hitting | sample_nfe256 | 256 | 256 | 512 | 0 | 0.9307957 | 0.04856185 | 13.62993 | 4.364337 | 284.6 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | bd3lm_ancestral | sample_nfe1 | 1 | 16 | 512 | 0 | 0.00458246 | 0.9809891 | 650.0576 | 4.472699 | 251.9 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | bd3lm_ancestral | sample_nfe2 | 2 | 32 | 512 | 0 | 0.008704328 | 0.876842 | 237.1465 | 4.470998 | 354.3 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | bd3lm_ancestral | sample_nfe4 | 4 | 64 | 512 | 0 | 0.2986233 | 0.2948406 | 73.2943 | 4.435491 | 282.2 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | bd3lm_ancestral | sample_nfe8 | 8 | 128 | 512 | 0 | 0.7792172 | 0.1025696 | 32.56643 | 4.391209 | 290.7 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | bd3lm_ancestral | sample_nfe16 | 16 | 256 | 512 | 0 | 0.9507253 | 0.04050174 | 21.5111 | 4.369975 | 278.6 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | bd3lm_ancestral | sample_nfe32 | 32 | 512 | 512 | 0 | 0.9360491 | 0.04614744 | 17.17839 | 4.369438 | 289.2 |
| MDLM | best_canonical_100k | MDLM_best_headline_n2048 | mdlm | sample_nfe64 | 64 | 65 | 2048 | 0 | 0.8183863 | 0.08981373 | 17.43788 | 4.35767 | 783.6 |
| MDLM | best_canonical_100k | MDLM_best_sweep | mdlm | sample_nfe16 | 16 | 17 | 512 | 0 | 0.9157371 | 0.05539063 | 26.99332 | 4.412504 | 281.5 |
| MDLM | best_canonical_100k | MDLM_best_sweep | mdlm | sample_nfe32 | 32 | 33 | 512 | 0 | 0.9418216 | 0.04431451 | 20.09906 | 4.364651 | 281.1 |
| MDLM | best_canonical_100k | MDLM_best_sweep | mdlm | sample_nfe64 | 64 | 65 | 512 | 0 | 0.9445849 | 0.04312114 | 17.13951 | 4.350361 | 287.3 |
| MDLM | best_canonical_100k | MDLM_best_sweep | mdlm | sample_nfe128 | 128 | 129 | 512 | 0 | 0.9157252 | 0.0547128 | 15.86025 | 4.338375 | 282.5 |
| MDLM | best_canonical_100k | MDLM_best_sweep | mdlm | sample_nfe256 | 256 | 257 | 512 | 0 | 0.9436481 | 0.04373464 | 15.40935 | 4.332745 | 298.3 |
| gold_reference | validation_reference | gold_tinystories_n2048 | gold | gold |  | 0 | 2048 | 0 | 1 | 0 | 6.468816 | 4.376747 | 1095.4 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | full_cfm | argmax_nfe1 | 1 | 1 | 512 | 0 | 0.006974877 | 0.9147611 | 87.53626 | 4.049144 | 232.8 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | full_cfm | argmax_nfe2 | 2 | 2 | 512 | 0 | 0.00812656 | 0.891088 | 83.03012 | 4.108866 | 210.5 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | full_cfm | argmax_nfe4 | 4 | 4 | 512 | 0 | 0.01340737 | 0.8104113 | 74.8903 | 4.261113 | 178.5 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | full_cfm | argmax_nfe8 | 8 | 8 | 512 | 0 | 0.03915727 | 0.636437 | 78.66352 | 4.419974 | 154 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | full_cfm | argmax_nfe16 | 16 | 16 | 512 | 0 | 0.1302309 | 0.439008 | 90.11573 | 4.534766 | 188 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | full_cfm | argmax_nfe32 | 32 | 32 | 512 | 0 | 0.1973022 | 0.3678088 | 91.93051 | 4.552089 | 183.1 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | full_cfm | argmax_nfe64 | 64 | 64 | 512 | 0 | 0.3440323 | 0.2689904 | 89.63575 | 4.552963 | 195.5 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | full_cfm | sample_nfe1 | 1 | 1 | 512 | 0 | 0.007023321 | 0.9139226 | 96.05345 | 4.085438 | 299.8 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | full_cfm | sample_nfe2 | 2 | 2 | 512 | 0 | 0.007091647 | 0.9125613 | 92.26585 | 4.152983 | 216 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | full_cfm | sample_nfe4 | 4 | 4 | 512 | 0 | 0.0113859 | 0.8364363 | 80.32333 | 4.288379 | 262.4 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | full_cfm | sample_nfe8 | 8 | 8 | 512 | 0 | 0.02601866 | 0.7032436 | 84.24898 | 4.430688 | 293.9 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | full_cfm | sample_nfe16 | 16 | 16 | 512 | 0 | 0.1490761 | 0.415322 | 100.716 | 4.540715 | 180 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | full_cfm | sample_nfe32 | 32 | 32 | 512 | 0 | 0.3212329 | 0.2822067 | 100.0524 | 4.554562 | 183.1 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | full_cfm | sample_nfe64 | 64 | 64 | 512 | 0 | 0.2802168 | 0.3066607 | 94.33208 | 4.554597 | 162.4 |
| AR | byte_trained_100k | ar_last | ar | argmax_ar | 256 | 256 | 512 | 0 | 0.004072096 | 1 |  | 3.075191 | 55.4 |
| AR | byte_trained_100k | ar_last | ar | sample_ar | 256 | 256 | 512 | 0 | 0.1295738 | 0.4386295 |  | 2.941333 | 32 |
| AR | byte_trained_100k | ar_last_clean | ar | argmax_ar | 256 | 256 | 512 | 0 | 0.004072096 | 1 |  | 3.474379 | 20.2 |
| AR | byte_trained_100k | ar_last_clean | ar | sample_ar | 256 | 256 | 512 | 0 | 0.1249503 | 0.4447265 |  | 3.542434 | 19.2 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | argmax_nofh_spb1 | 1 | 16 | 512 | 0 | 0.004072096 | 1 |  | 0 | 96.7 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | sample_nofh_spb1 | 1 | 16 | 512 | 0 | 0.004072096 | 1 |  | 3.50146 | 26.7 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | argmax_nofh_spb2 | 2 | 32 | 512 | 0 | 0.004363454 | 0.9886233 |  | 3.036466 | 55.2 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | sample_nofh_spb2 | 2 | 32 | 512 | 0 | 0.004332722 | 0.989841 |  | 3.616308 | 72.9 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | argmax_nofh_spb4 | 4 | 64 | 512 | 0 | 0.007222544 | 0.906551 |  | 3.547232 | 27.1 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | sample_nofh_spb4 | 4 | 64 | 512 | 0 | 0.006448225 | 0.9248217 |  | 3.682347 | 50.1 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | argmax_nofh_spb8 | 8 | 128 | 512 | 0 | 0.03036192 | 0.675442 |  | 3.550849 | 83.1 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | sample_nofh_spb8 | 8 | 128 | 512 | 0 | 0.02284755 | 0.7195228 |  | 3.661238 | 28.4 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | argmax_nofh_spb16 | 16 | 256 | 512 | 0 | 0.05208247 | 0.5879609 |  | 3.529267 | 38.8 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | sample_nofh_spb16 | 16 | 256 | 512 | 0 | 0.06591139 | 0.5499371 |  | 3.621997 | 90.3 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | argmax_nofh_spb32 | 32 | 512 | 512 | 0 | 0.05014627 | 0.5936442 |  | 3.508096 | 28.8 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | sample_nofh_spb32 | 32 | 512 | 512 | 0 | 0.08037503 | 0.5178028 |  | 3.610518 | 49.2 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | argmax_fh_spb16 | 16 | 256 | 512 | 0 | 0.08623292 | 0.5062525 |  | 3.4933 | 84.3 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | bd3lm | sample_fh_spb16 | 16 | 256 | 512 | 0 | 0.1093303 | 0.4662294 |  | 3.591062 | 22.1 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | argmax_nofh_spb1 | 1 | 16 | 512 | 0 | 0.004072096 | 1 |  | 0 | 34.6 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | sample_nofh_spb1 | 1 | 16 | 512 | 0 | 0.004072096 | 1 |  | 3.50146 | 96.4 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | argmax_nofh_spb2 | 2 | 32 | 512 | 0 | 0.004363454 | 0.9886233 |  | 3.036402 | 63.7 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | sample_nofh_spb2 | 2 | 32 | 512 | 0 | 0.004332722 | 0.989841 |  | 3.616052 | 95.1 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | argmax_nofh_spb4 | 4 | 64 | 512 | 0 | 0.008592138 | 0.8790166 |  | 3.546191 | 27.8 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | sample_nofh_spb4 | 4 | 64 | 512 | 0 | 0.006698032 | 0.9184933 |  | 3.681494 | 26.1 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | argmax_nofh_spb8 | 8 | 128 | 512 | 0 | 0.03176752 | 0.667511 |  | 3.548509 | 22.3 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | sample_nofh_spb8 | 8 | 128 | 512 | 0 | 0.02033345 | 0.7385074 |  | 3.657862 | 85.5 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | argmax_nofh_spb16 | 16 | 256 | 512 | 0 | 0.05198447 | 0.5880804 |  | 3.525822 | 52.3 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | sample_nofh_spb16 | 16 | 256 | 512 | 0 | 0.07340262 | 0.5307763 |  | 3.617768 | 89.1 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | argmax_nofh_spb32 | 32 | 512 | 512 | 0 | 0.04971004 | 0.5950989 |  | 3.503683 | 28.5 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | sample_nofh_spb32 | 32 | 512 | 512 | 0 | 0.08303985 | 0.5123822 |  | 3.605866 | 25.6 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | argmax_fh_spb16 | 16 | 256 | 512 | 0 | 0.08299379 | 0.5124828 |  | 3.489248 | 28.1 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | bd3lm | sample_fh_spb16 | 16 | 256 | 512 | 0 | 0.09926685 | 0.4830831 |  | 3.585372 | 75.3 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | full_cfm | argmax_nfe1 | 1 | 1 | 512 | 0 | 0.005743826 | 0.9460265 | 74.13102 |  | 256 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | full_cfm | argmax_nfe2 | 2 | 2 | 512 | 0 | 0.005651912 | 0.9486856 | 76.19676 |  | 246.3 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | full_cfm | argmax_nfe4 | 4 | 4 | 512 | 0 | 0.006439182 | 0.9278842 | 92.30155 |  | 247.9 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | full_cfm | argmax_nfe8 | 8 | 8 | 512 | 0 | 0.006969371 | 0.9143406 | 107.8877 |  | 247.1 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | full_cfm | argmax_nfe16 | 16 | 16 | 512 | 0 | 0.008014951 | 0.8919696 | 116.753 |  | 247.8 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | full_cfm | argmax_nfe32 | 32 | 32 | 512 | 0 | 0.007412399 | 0.9048873 | 126.3127 |  | 247.9 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | full_cfm | argmax_nfe64 | 64 | 64 | 512 | 0 | 0.008715979 | 0.8781743 | 135.9142 |  | 248.6 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | full_cfm | argmax_nfe128 | 128 | 128 | 512 | 0 | 0.01040985 | 0.8506087 | 143.2454 |  | 288.5 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | full_cfm | argmax_nfe256 | 256 | 256 | 512 | 0 | 0.01020356 | 0.8530334 | 149.9478 |  | 247.3 |

## External generation metrics: distribution diagnostics

| model_family | model_variant | evaluation_label | point | seq_rep_2 | seq_rep_3 | seq_rep_4 | distinct_2 | distinct_3 | distinct_4 | cross_sample_overlap_4 | zipf | js_1gram | js_2gram |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BD3-LM | best_canonical_100k | BD3LM_best_headline_n2048 | sample_nfe4 | 0.09903684 | 0.01929864 | 0.005328635 | 0.2296952 | 0.6231853 | 0.8569066 | 0.003819496 | 1.755912 | 0.01890683 | 0.1740389 |
| BD3-LM | best_canonical_100k | BD3LM_best_native | sample_nfe256 | 0.1625996 | 0.05201156 | 0.02124506 | 0.2327972 | 0.5397084 | 0.760692 | 0.01029727 | 1.539833 | 0.02065817 | 0.138761 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | sample_nfe1 | 0.03572304 | 0.001291831 | 0.0001003582 | 0.4851333 | 0.9130859 | 0.9918324 | 9.282178e-05 | 1.609938 | 0.04133715 | 0.4853205 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | sample_nfe2 | 0.05854779 | 0.006105438 | 0.0009727026 | 0.4296186 | 0.8540231 | 0.9670285 | 0.001086618 | 1.536368 | 0.03490815 | 0.3046182 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | sample_nfe4 | 0.09903493 | 0.01904681 | 0.005187747 | 0.3537607 | 0.7426104 | 0.9102334 | 0.003819496 | 1.520328 | 0.02507551 | 0.2086751 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | sample_nfe8 | 0.1310432 | 0.03372601 | 0.01161839 | 0.2973422 | 0.6518362 | 0.8500262 | 0.007168556 | 1.525581 | 0.02158841 | 0.167478 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | sample_nfe16 | 0.1464844 | 0.04171537 | 0.01521585 | 0.2655331 | 0.5968796 | 0.8083467 | 0.007976791 | 1.522462 | 0.02062944 | 0.1518967 |
| BD3-LM | best_canonical_100k | BD3LM_best_sweep | sample_nfe32 | 0.1554534 | 0.04699803 | 0.01829607 | 0.2497319 | 0.5702971 | 0.7863837 | 0.009233531 | 1.529885 | 0.02021612 | 0.1432133 |
| MDLM | best_canonical_100k | MDLM_best_headline_n2048 | sample_nfe64 | 0.1607441 | 0.04912225 | 0.01836169 | 0.1328527 | 0.3978877 | 0.6525117 | 0.008912609 | 1.811147 | 0.01667773 | 0.1065631 |
| MDLM | best_canonical_100k | MDLM_best_sweep | sample_nfe16 | 0.1321768 | 0.0332954 | 0.01016706 | 0.2553615 | 0.5924043 | 0.8131485 | 0.005748213 | 1.592814 | 0.02358735 | 0.1514783 |
| MDLM | best_canonical_100k | MDLM_best_sweep | sample_nfe32 | 0.1557368 | 0.04530635 | 0.01653594 | 0.238534 | 0.5577095 | 0.7813967 | 0.007745724 | 1.587269 | 0.02247273 | 0.1426001 |
| MDLM | best_canonical_100k | MDLM_best_sweep | sample_nfe64 | 0.1633578 | 0.05024299 | 0.0189291 | 0.2226409 | 0.5309501 | 0.7569402 | 0.008912609 | 1.589963 | 0.0241778 | 0.1412929 |
| MDLM | best_canonical_100k | MDLM_best_sweep | sample_nfe128 | 0.1693168 | 0.05451064 | 0.02259604 | 0.2164369 | 0.5175858 | 0.7438936 | 0.009748945 | 1.588939 | 0.02379346 | 0.1388366 |
| MDLM | best_canonical_100k | MDLM_best_sweep | sample_nfe256 | 0.1706419 | 0.05491049 | 0.02197073 | 0.2148055 | 0.515502 | 0.7427047 | 0.01027491 | 1.59347 | 0.02405893 | 0.138917 |
| gold_reference | validation_reference | gold_tinystories_n2048 | gold | 0.1709463 | 0.06695412 | 0.02936056 | 0.1630323 | 0.4365004 | 0.6649484 | 0.01241551 | 1.683027 | 0 | 0 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | argmax_nfe1 | 0.1418199 | 0.03287248 | 0.00893188 | 0.1153876 | 0.5230376 | 0.8281018 | 0.006284654 | 2.270608 | 0.1180765 | 0.3266114 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | argmax_nfe2 | 0.1369715 | 0.03324926 | 0.009680707 | 0.1271293 | 0.5354023 | 0.8241493 | 0.006851466 | 2.294799 | 0.1058265 | 0.3020371 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | argmax_nfe4 | 0.1233073 | 0.03249569 | 0.01015934 | 0.1618183 | 0.5690284 | 0.828511 | 0.007451357 | 2.202428 | 0.08593922 | 0.2587634 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | argmax_nfe8 | 0.1058594 | 0.02998893 | 0.01039093 | 0.2289369 | 0.6279681 | 0.847139 | 0.008294465 | 1.970297 | 0.05951619 | 0.221487 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | argmax_nfe16 | 0.09509038 | 0.0287663 | 0.01009758 | 0.3137638 | 0.6882689 | 0.8668555 | 0.008046977 | 1.590805 | 0.03803636 | 0.2095363 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | argmax_nfe32 | 0.09507506 | 0.02922767 | 0.01054533 | 0.334712 | 0.6965966 | 0.8676661 | 0.008324019 | 1.470009 | 0.03969959 | 0.2137026 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base | argmax_nfe64 | 0.09636183 | 0.02988127 | 0.01076149 | 0.3368949 | 0.6947742 | 0.8658906 | 0.008594925 | 1.450134 | 0.04099522 | 0.2138189 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | sample_nfe1 | 0.1333104 | 0.03010427 | 0.007750741 | 0.13269 | 0.5513503 | 0.8446146 | 0.005134949 | 2.364817 | 0.1088883 | 0.3208905 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | sample_nfe2 | 0.1284314 | 0.0308117 | 0.008553607 | 0.1537071 | 0.5719888 | 0.8433872 | 0.006232072 | 2.099567 | 0.09252514 | 0.2933947 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | sample_nfe4 | 0.1190947 | 0.0313961 | 0.009804224 | 0.1825827 | 0.5907896 | 0.8403764 | 0.0068393 | 2.067918 | 0.07624343 | 0.2524568 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | sample_nfe8 | 0.1046569 | 0.02981976 | 0.01022882 | 0.2463771 | 0.6442929 | 0.8548203 | 0.007146617 | 1.906773 | 0.05217171 | 0.2202014 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | sample_nfe16 | 0.09450061 | 0.02867403 | 0.01005126 | 0.3396446 | 0.7042784 | 0.8738266 | 0.006530425 | 1.489488 | 0.03540193 | 0.2131056 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | sample_nfe32 | 0.09541207 | 0.02939684 | 0.01048357 | 0.3524893 | 0.7051858 | 0.871611 | 0.007096822 | 1.398543 | 0.03999789 | 0.2166467 |
| CFM-M1 | full_sequence_best_step68001 | m1_tinystories_base_sample | sample_nfe64 | 0.09666054 | 0.02991203 | 0.01069201 | 0.3468367 | 0.699065 | 0.8677356 | 0.007640353 | 1.421028 | 0.04128587 | 0.215215 |
| AR | byte_trained_100k | ar_last | argmax_ar | 0.5215686 | 0.3149606 | 0.229249 | 0.0009344363 | 0.001337968 | 0.001505373 | 1 | 1.14607 | 0.6911193 | 0.6931472 |
| AR | byte_trained_100k | ar_last | sample_ar | 0.5097426 | 0.273822 | 0.1736351 | 0.005154718 | 0.0270131 | 0.07882751 | 0.08499028 | 2.543028 | 0.6905966 | 0.6931472 |
| AR | byte_trained_100k | ar_last_clean | argmax_ar | 0.04166667 | 0 | 0 | 0.001871745 | 0.001953125 | 0.001953125 | 1 | 0.4191824 | 0.6931472 | 0.6931472 |
| AR | byte_trained_100k | ar_last_clean | sample_ar | 0.04409336 | 0.01606925 | 0.007116345 | 0.4299438 | 0.6988104 | 0.8322375 | 0.006162153 | 1.149525 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | argmax_nofh_spb1 |  |  |  |  |  |  |  |  |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | sample_nofh_spb1 | 0 | 0 | 0 | 0.989002 | 0.99994 | 1 | 0 | 0.2629596 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | argmax_nofh_spb2 | 0.0287434 | 0.003245194 | 0.0001994087 | 0.6214664 | 0.9182419 | 0.989602 | 0 | 0.9213707 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | sample_nofh_spb2 | 0.0002007989 | 0 | 0 | 0.9823334 | 0.9998951 | 1 | 0 | 0.3194418 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | argmax_nofh_spb4 | 0.03676621 | 0.005115444 | 0.0006342291 | 0.6102508 | 0.8890137 | 0.9745432 | 0.0004111842 | 0.8668912 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | sample_nofh_spb4 | 0.001446718 | 8.231893e-05 | 0 | 0.9225274 | 0.9942708 | 0.9984761 | 0 | 0.4813377 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | argmax_nofh_spb8 | 0.04066183 | 0.007820012 | 0.001495902 | 0.516079 | 0.805626 | 0.9270697 | 0.0005354954 | 1.011209 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | sample_nofh_spb8 | 0.005867519 | 0.0002927407 | 0 | 0.8033499 | 0.9667461 | 0.9892285 | 8.877841e-05 | 0.6786665 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | argmax_nofh_spb16 | 0.04927053 | 0.01229911 | 0.003534941 | 0.4492865 | 0.7383494 | 0.8806445 | 0.002080797 | 1.107365 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | sample_nofh_spb16 | 0.01246493 | 0.001366331 | 3.829657e-05 | 0.6969987 | 0.922588 | 0.9749917 | 0.0002663352 | 0.8255536 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | argmax_nofh_spb32 | 0.05581762 | 0.01746825 | 0.005964566 | 0.4088369 | 0.6931085 | 0.8431863 | 0.0005266271 | 1.174018 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | sample_nofh_spb32 | 0.01510499 | 0.00179792 | 0.0002000954 | 0.6447829 | 0.902149 | 0.9683386 | 0.0003591954 | 0.9173543 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | argmax_fh_spb16 | 0.06582685 | 0.02202236 | 0.008532167 | 0.3609448 | 0.6375932 | 0.8049174 | 0.003066825 | 1.243321 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last | sample_fh_spb16 | 0.02176101 | 0.003865049 | 0.0007231915 | 0.5772037 | 0.8556241 | 0.9452912 | 0.003089328 | 1.008692 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | argmax_nofh_spb1 |  |  |  |  |  |  |  |  |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | sample_nofh_spb1 | 0 | 0 | 0 | 0.989002 | 0.99994 | 1 | 0 | 0.2629596 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | argmax_nofh_spb2 | 0.0287434 | 0.003245194 | 0.0001994087 | 0.6214451 | 0.9182372 | 0.9896014 | 0 | 0.9215118 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | sample_nofh_spb2 | 0.0002007989 | 0 | 0 | 0.9823289 | 0.9998951 | 1 | 0 | 0.3192044 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | argmax_nofh_spb4 | 0.03678338 | 0.005116863 | 0.0006342291 | 0.6105339 | 0.8894276 | 0.9749902 | 0.0003324468 | 0.8662887 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | sample_nofh_spb4 | 0.001448608 | 8.231893e-05 | 0 | 0.9227341 | 0.9945824 | 0.9987522 | 0 | 0.48082 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | argmax_nofh_spb8 | 0.04071052 | 0.00783022 | 0.001496733 | 0.5167016 | 0.8066667 | 0.9282604 | 0.0005362846 | 1.010135 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | sample_nofh_spb8 | 0.005880999 | 0.0002948545 | 0 | 0.804821 | 0.9683305 | 0.9905693 | 0 | 0.6775335 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | argmax_nofh_spb16 | 0.04934519 | 0.0123187 | 0.003541008 | 0.4503915 | 0.7397659 | 0.8822121 | 0.00210003 | 1.105958 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | sample_nofh_spb16 | 0.01249939 | 0.001368915 | 3.829657e-05 | 0.6988468 | 0.9248222 | 0.9770772 | 0.0001795977 | 0.8241855 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | argmax_nofh_spb32 | 0.05590943 | 0.01749179 | 0.005972334 | 0.4101707 | 0.6950583 | 0.8454237 | 0.0005274489 | 1.172338 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | sample_nofh_spb32 | 0.01513554 | 0.001802343 | 0.0002000954 | 0.6467707 | 0.904568 | 0.9706476 | 0.0003676471 | 0.9158379 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | argmax_fh_spb16 | 0.06592899 | 0.02204498 | 0.00854177 | 0.3619987 | 0.6392352 | 0.8068891 | 0.00298899 | 1.2417 | 0.6931472 | 0.6931472 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_last_clean | sample_fh_spb16 | 0.02181597 | 0.003872239 | 0.0007231915 | 0.579362 | 0.8583427 | 0.948228 | 0.002837629 | 1.006909 | 0.6931472 | 0.6931472 |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | argmax_nfe1 |  |  |  |  |  |  |  |  |  |  |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | argmax_nfe2 |  |  |  |  |  |  |  |  |  |  |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | argmax_nfe4 |  |  |  |  |  |  |  |  |  |  |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | argmax_nfe8 |  |  |  |  |  |  |  |  |  |  |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | argmax_nfe16 |  |  |  |  |  |  |  |  |  |  |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | argmax_nfe32 |  |  |  |  |  |  |  |  |  |  |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | argmax_nfe64 |  |  |  |  |  |  |  |  |  |  |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | argmax_nfe128 |  |  |  |  |  |  |  |  |  |  |
| ECLD | lr1e-4_fixed_reduction | ECLD_last_step20000_argmax | argmax_nfe256 |  |  |  |  |  |  |  |  |  |  |

## Sequence latency benchmarks

| model_family | model_variant | evaluation_label | sampler | point | nfe | forwards_per_sequence | sequence_latency_ms | sequence_latency_p10_ms | sequence_latency_p90_ms | sequence_latency_repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| MDLM | best_canonical_100k | mdlm_nfe16 | mdlm | sample_nfe16 | 16 | 17 | 195.4245 | 193.1746 | 198.4278 | 30 |
| MDLM | best_canonical_100k | mdlm_nfe32 | mdlm | sample_nfe32 | 32 | 33 | 380.1811 | 378.0244 | 388.6836 | 30 |
| MDLM | best_canonical_100k | mdlm_nfe64 | mdlm | sample_nfe64 | 64 | 65 | 746.5357 | 733.7207 | 753.9503 | 30 |
| MDLM | best_canonical_100k | mdlm_nfe128 | mdlm | sample_nfe128 | 128 | 129 | 1384.921 | 1352.262 | 1419.356 | 30 |
| MDLM | best_canonical_100k | mdlm_nfe256 | mdlm | sample_nfe256 | 256 | 257 | 2315.566 | 2261.444 | 2359.78 | 30 |
| BD3-LM | best_canonical_100k | bd3lm_ancestral_b16_spb1 | bd3lm_ancestral | sample_nfe1 | 1 | 32 | 244.0762 | 239.6572 | 276.3495 | 30 |
| BD3-LM | best_canonical_100k | bd3lm_ancestral_b16_spb2 | bd3lm_ancestral | sample_nfe2 | 2 | 48 | 360.1427 | 354.4587 | 372.5024 | 30 |
| BD3-LM | best_canonical_100k | bd3lm_ancestral_b16_spb4 | bd3lm_ancestral | sample_nfe4 | 4 | 80 | 608.3198 | 592.0558 | 705.2767 | 30 |
| BD3-LM | best_canonical_100k | bd3lm_ancestral_b16_spb8 | bd3lm_ancestral | sample_nfe8 | 8 | 144 | 1085.801 | 1065.116 | 1140.409 | 30 |
| BD3-LM | best_canonical_100k | bd3lm_ancestral_b16_spb16 | bd3lm_ancestral | sample_nfe16 | 16 | 272 | 2081.987 | 2059.44 | 2161.965 | 30 |
| BD3-LM | best_canonical_100k | bd3lm_ancestral_b16_spb32 | bd3lm_ancestral | sample_nfe32 | 32 | 528 | 4037.253 | 3970.601 | 4225.22 | 30 |
| BD3-LM | best_canonical_100k | bd3lm_first_hitting_b16_spb256 | bd3lm_first_hitting | sample_nfe256 | 256 | 272 | 2182.74 | 2157.874 | 2345.233 | 30 |

## Online quality curves

| model_family | model_variant | training_step | evaluation_label | point | nfe | validation_event | mauve | frontier_integral | gen_ppl | token_entropy | sampling_seconds | mauve_seconds | gen_ppl_seconds | total_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ECLD | lr1e-4_fixed_reduction | 4000 | ecld1e4_online_step4000 | online_quality_nfe1 | 1 | 1 | 0.006144286 | 0.9343686 | 111.8483 | 3.576602 | 0.7158819 | 53.23613 | 75.15368 | 129.8431 |
| ECLD | lr1e-4_fixed_reduction | 8000 | ecld1e4_online_step8000 | online_quality_nfe1 | 1 | 2 | 0.006463377 | 0.9273731 | 91.11767 | 3.812846 | 0.6427671 | 47.17184 | 74.43684 | 122.7986 |
| ECLD | lr1e-4_fixed_reduction | 12000 | ecld1e4_online_step12000 | online_quality_nfe1 | 1 | 3 | 0.005333718 | 0.9567988 | 77.72694 | 3.528897 | 0.6456572 | 48.96037 | 74.79881 | 125.0432 |
| ECLD | lr1e-4_fixed_reduction | 16000 | ecld1e4_online_step16000 | online_quality_nfe1 | 1 | 4 | 0.006022001 | 0.9377147 | 91.54879 | 3.388227 | 0.6444082 | 46.31003 | 75.14033 | 122.7409 |
| ECLD | lr1e-4_fixed_reduction | 20000 | ecld1e4_online_step20000 | online_quality_nfe1 | 1 | 5 | 0.007229547 | 0.9095495 | 75.20935 | 3.294851 | 0.6492388 | 52.31703 | 74.52659 | 128.1082 |
| ECLD | lr1e-4_fixed_reduction | 24000 | ecld1e4_online_step24000 | online_quality_nfe1 | 1 | 6 | 0.005328173 | 0.9574935 | 89.66468 | 3.452966 | 0.6534339 | 49.08232 | 74.84474 | 125.2165 |
| ECLD | lr1e-4_fixed_reduction | 28000 | ecld1e4_online_step28000 | online_quality_nfe1 | 1 | 7 | 0.005915156 | 0.941025 | 115.971 | 3.819479 | 0.6490599 | 47.11686 | 75.45786 | 123.8561 |
| ECLD | lr1e-4_fixed_reduction | 32000 | ecld1e4_online_step32000 | online_quality_nfe1 | 1 | 8 | 0.006337527 | 0.9299027 | 84.76593 | 3.409616 | 0.6456786 | 50.65798 | 74.68171 | 126.63 |
| ECLD | lr1e-4_fixed_reduction | 36000 | ecld1e4_online_step36000 | online_quality_nfe1 | 1 | 9 | 0.005253199 | 0.9594804 | 58.09486 | 3.050646 | 0.6409334 | 47.57269 | 74.88576 | 123.7138 |
| ECLD | lr1e-4_fixed_reduction | 40000 | ecld1e4_online_step40000 | online_quality_nfe1 | 1 | 10 | 0.00640488 | 0.9282855 | 106.4675 | 3.721201 | 0.6475654 | 40.92092 | 74.92213 | 117.1179 |
| ECLD | lr1e-4_fixed_reduction | 44000 | ecld1e4_online_step44000 | online_quality_nfe1 | 1 | 11 | 0.006063289 | 0.937333 | 92.84736 | 3.572418 | 0.6470121 | 46.43915 | 74.67909 | 122.4067 |
| ECLD | lr1e-4_fixed_reduction | 48000 | ecld1e4_online_step48000 | online_quality_nfe1 | 1 | 12 | 0.006430872 | 0.9276939 | 100.3361 | 3.647584 | 0.654024 | 32.61419 | 74.90807 | 108.809 |
| ECLD | lr1e-4_fixed_reduction | 52000 | ecld1e4_online_step52000 | online_quality_nfe1 | 1 | 13 | 0.00533826 | 0.9574033 | 99.03862 | 3.723786 | 0.6467448 | 42.66002 | 74.97459 | 118.8957 |
| ECLD | lr1e-4_fixed_reduction | 56000 | ecld1e4_online_step56000 | online_quality_nfe1 | 1 | 14 | 0.006032415 | 0.9383145 | 100.4564 | 3.446863 | 0.6467769 | 34.36776 | 74.68169 | 110.3188 |
| ECLD | lr1e-4_fixed_reduction | 60000 | ecld1e4_online_step60000 | online_quality_nfe1 | 1 | 15 | 0.005124658 | 0.9635923 | 132.5499 | 3.738638 | 0.6425617 | 38.24087 | 74.93625 | 114.4628 |
| ECLD | lr1e-4_fixed_reduction | 64000 | ecld1e4_online_step64000 | online_quality_nfe1 | 1 | 16 | 0.005673469 | 0.9470598 | 99.25369 | 3.494263 | 0.6408831 | 48.25807 | 74.82314 | 124.3504 |
| ECLD | lr1e-4_fixed_reduction | 68000 | ecld1e4_online_step68000 | online_quality_nfe1 | 1 | 17 | 0.005970443 | 0.9399412 | 99.05253 | 3.632554 | 1.377151 | 68.73149 | 93.90673 | 164.6462 |
| ECLD | lr1e-4_fixed_reduction | 72000 | ecld1e4_online_step72000 | online_quality_nfe1 | 1 | 18 | 0.006689745 | 0.9213171 | 108.4486 | 3.699607 | 1.397995 | 86.32962 | 126.4027 | 215.4012 |
| ECLD | lr1e-4_fixed_reduction | 76000 | ecld1e4_online_step76000 | online_quality_nfe1 | 1 | 19 | 0.005495019 | 0.9526644 | 107.0557 | 3.656411 | 0.6443322 | 47.80564 | 75.02013 | 124.1435 |
| ECLD | lr1e-4_fixed_reduction | 80000 | ecld1e4_online_step80000 | online_quality_nfe1 | 1 | 20 | 0.00561092 | 0.9489732 | 121.9481 | 3.704925 | 0.6463797 | 48.73979 | 75.31896 | 125.3404 |
| ECLD | lr1e-4_fixed_reduction | 84000 | ecld1e4_online_step84000 | online_quality_nfe1 | 1 | 21 | 0.005679458 | 0.9477855 | 96.61124 | 3.382324 | 0.6402257 | 47.11721 | 74.71941 | 123.1181 |
| ECLD | lr1e-4_fixed_reduction | 88000 | ecld1e4_online_step88000 | online_quality_nfe1 | 1 | 22 | 0.006142002 | 0.9353979 | 94.77174 | 3.556221 | 0.6419922 | 46.38523 | 74.84136 | 122.5032 |
| ECLD | lr1e-4_fixed_reduction | 92000 | ecld1e4_online_step92000 | online_quality_nfe1 | 1 | 23 | 0.005093743 | 0.9645896 | 94.68684 | 3.58047 | 0.6441771 | 50.73734 | 75.13586 | 127.1445 |
| ECLD | lr1e-4_fixed_reduction | 96000 | ecld1e4_online_step96000 | online_quality_nfe1 | 1 | 24 | 0.005689131 | 0.9470745 | 102.3532 | 3.58683 | 0.648028 | 49.14005 | 74.70461 | 125.177 |
| ECLD | lr1e-4_fixed_reduction | 100000 | ecld1e4_online_step100000 | online_quality_nfe1 | 1 | 25 | 0.00629458 | 0.9312522 | 116.1591 | 3.763873 | 0.646357 | 49.62297 | 74.62471 | 125.5484 |
| ECLD | lr1e-3_failed_ablation | 4000 | ecld1e3_online_step4000 | online_quality_nfe1 | 1 | 1 | 0.004072096 | 1 | 6.038054 | 0 | 0.6936096 | 44.86292 | 31.71527 | 77.94885 |
| ECLD | lr1e-3_failed_ablation | 8000 | ecld1e3_online_step8000 | online_quality_nfe1 | 1 | 2 | 0.004072096 | 1 | 6.232091 | 0 | 0.6188755 | 41.52893 | 30.2871 | 72.92283 |
| ECLD | lr1e-3_failed_ablation | 12000 | ecld1e3_online_step12000 | online_quality_nfe1 | 1 | 3 | 0.004072096 | 1 | 6.351632 | 0.1012492 | 0.6193691 | 44.87066 | 30.42594 | 76.50031 |
| ECLD | lr1e-3_failed_ablation | 16000 | ecld1e3_online_step16000 | online_quality_nfe1 | 1 | 4 | 0.004072096 | 1 | 14.18182 | 0.006738565 | 0.6240188 | 39.0089 | 30.49469 | 70.71911 |

## Validation curves

| model_family | model_variant | training_step | epoch | val_loss | val_nll | val_nelbo | val_bits_per_character | val_bits_per_character_estimate | val_masked_fraction | val_token_count | val_schedule_candidates | val_selected_mask_rate_min | val_selected_mask_rate_max | val_selected_schedule_nelbo_variance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | byte_trained_100k | 10000 |  | 0.9340693 | 0.9340693 |  | 1.347577 |  |  | 1638400 |  |  |  |  |
| AR | byte_trained_100k | 20000 |  | 0.4655987 | 0.4655987 |  | 0.6717169 |  |  | 1638400 |  |  |  |  |
| AR | byte_trained_100k | 30000 |  | 0.4182391 | 0.4182391 |  | 0.6033915 |  |  | 1638400 |  |  |  |  |
| AR | byte_trained_100k | 40000 |  | 0.3987342 | 0.3987342 |  | 0.5752518 |  |  | 1638400 |  |  |  |  |
| AR | byte_trained_100k | 50000 |  | 0.3872092 | 0.3872092 |  | 0.5586248 |  |  | 1638400 |  |  |  |  |
| AR | byte_trained_100k | 60000 |  | 0.3800081 | 0.3800081 |  | 0.5482358 |  |  | 1638400 |  |  |  |  |
| AR | byte_trained_100k | 70000 |  | 0.3745951 | 0.3745951 |  | 0.5404265 |  |  | 1638400 |  |  |  |  |
| AR | byte_trained_100k | 80000 |  | 0.3714371 | 0.3714371 |  | 0.5358704 |  |  | 1638400 |  |  |  |  |
| AR | byte_trained_100k | 90000 |  | 0.3693286 | 0.3693286 |  | 0.5328285 |  |  | 1638400 |  |  |  |  |
| AR | byte_trained_100k | 100000 |  | 0.3683956 | 0.3683956 |  | 0.5314824 |  |  | 1638400 |  |  |  |  |
| BD3-LM | byte_block16_trained_100k | 10000 |  | 0.5909288 |  | 0.5909288 |  | 0.85253 | 0.5004388 | 1638400 | 36 | 0.05 | 0.55 | 21.2798 |
| BD3-LM | byte_block16_trained_100k | 20000 |  | 0.5792629 |  | 0.5792629 |  | 0.8356998 | 0.500838 | 1638400 | 36 | 0.05 | 0.55 | 19.4656 |
| BD3-LM | byte_block16_trained_100k | 30000 |  | 0.5775812 |  | 0.5775812 |  | 0.8332735 | 0.5003595 | 1638400 | 36 | 0.05 | 0.55 | 18.14499 |
| BD3-LM | byte_block16_trained_100k | 40000 |  | 0.5715034 |  | 0.5715034 |  | 0.8245051 | 0.5000543 | 1638400 | 36 | 0.05 | 0.55 | 17.70323 |
| BD3-LM | byte_block16_trained_100k | 50000 |  | 0.5670424 |  | 0.5670424 |  | 0.8180693 | 0.5003027 | 1638400 | 36 | 0.05 | 0.55 | 17.01369 |
| BD3-LM | byte_block16_trained_100k | 60000 |  | 0.5659277 |  | 0.5659277 |  | 0.8164611 | 0.500528 | 1638400 | 36 | 0.05 | 0.55 | 16.25211 |
| BD3-LM | byte_block16_trained_100k | 70000 |  | 0.5614698 |  | 0.5614698 |  | 0.8100296 | 0.5008472 | 1638400 | 36 | 0.05 | 0.55 | 15.95916 |
| BD3-LM | byte_block16_trained_100k | 80000 |  | 0.5592797 |  | 0.5592797 |  | 0.80687 | 0.500993 | 1638400 | 36 | 0.05 | 0.55 | 15.80839 |
| BD3-LM | byte_block16_trained_100k | 90000 |  | 0.5524262 |  | 0.5524262 |  | 0.7969825 | 0.5004388 | 1638400 | 36 | 0.05 | 0.55 | 14.99949 |
| BD3-LM | byte_block16_trained_100k | 100000 |  | 0.5457352 |  | 0.5457352 |  | 0.7873295 | 0.5004431 | 1638400 | 36 | 0.05 | 0.55 | 14.98389 |

## Checkpoint metadata

| model_family | model_variant | model_status | run_id | checkpoint_path | checkpoint_step_x | checkpoint_step_y | checkpoint_global_step | checkpoint_epoch | checkpoint_loadable | checkpoint_best_metric | checkpoint_ema_decay | checkpoint_parameter_count_total | checkpoint_parameter_count_trainable | checkpoint_processed_tokens | checkpoint_wall_clock_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MDLM | trained_100k | complete_100k_full_eval | mdlm_tinystories_gpt2_h384_l6_seed12345 | baseline/MDLM_best.pt | 100000 | 100000 |  |  | 1 | 1.973708 | 0.9999 | 5.111944e+07 | 5.111944e+07 | 3.2768e+09 | 29842.7 |
| BD3-LM | block16_trained_100k | complete_100k_full_eval | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | baseline/BD3LM_best.pt | 100000 | 100000 |  |  | 1 | 1.813493 | 0.9999 | 5.111944e+07 | 5.111944e+07 | 3.2768e+09 | 57792.95 |
| AR | byte_trained_100k | complete_100k_legacy_byte | ar_tinystories_seed12345 | baseline/ar_tinystories_seed12345/checkpoints/last.pt | 100000 | 100000 |  |  | 1 | 0.3683956 | 0.9999 | 9.259674e+07 | 9.259674e+07 | 6.5536e+09 | 43547.79 |
| BD3-LM | byte_block16_trained_100k | complete_100k_legacy_byte | bd3lm_b16_tinystories_seed12345 | baseline/bd3lm_b16_tinystories_seed12345/checkpoints/last.pt | 100000 | 100000 |  |  | 1 | 0.5457352 | 0.9999 | 9.259674e+07 | 9.259674e+07 | 6.5536e+09 | 106901.5 |
| CFM-M1 | full_sequence_best_step68001 | best_checkpoint_available | tinystories_a100 | baseline/tinystories/NEW/tinystories_a100/checkpoints/best.ckpt | 68001 |  | 68001 | 4 | 1 |  |  |  |  |  |  |
| BCFM-M3 | block16_partial_step40000 | incomplete_40k_of_100k_no_final_eval | 2026-07-28_12-43-24_12345_local | logs/train/2026-07-28_12-43-24_12345_local/checkpoints/last.ckpt | 40000 |  | 40000 | 2 | 1 |  |  |  |  |  |  |
| ECLD | lr1e-3_failed_ablation | failed_degenerate_partial_progress16260_checkpoint8k | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | logs/train/2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345/checkpoints/best.ckpt | 8000 |  | 8000 | 0 | 1 |  |  |  |  |  |  |
| ECLD | lr1e-4_fixed_reduction | run_completed_100k_only_step20k_weights_saved | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | results/tinystories_ecld_last_step20000_eval_20260802/checkpoint/last-step00020000.ckpt | 20000 |  | 20000 | 1 | 1 |  |  |  |  |  |  |
| AR | gpt2_artifact_corrupt | checkpoint_corrupt_not_evaluable | AR_last_flat_artifact | baseline/AR_last.pt |  |  |  |  | 0 |  |  |  |  |  |  |

## Internal checkpoint evaluations

| model_family | model_variant | training_step | test_loss | test_nll | test_nelbo | test_perplexity | test_bits_per_character | test_bits_per_character_estimate | test_epsilon_truncated_nelbo_perplexity_estimate | test_masked_fraction | test_token_count | likelihood_is_strict_upper_bound | likelihood_time_min | processed_training_tokens | sample_adjacent_repetition_rate | sample_distinct_1 | sample_distinct_2 | sample_unigram_entropy_nats | sample_vocabulary_coverage | sampling_seconds | sampling_tokens_per_second |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | byte_trained_100k | 100000 | 0.5232632 | 0.5232632 |  | 1.687525 | 0.7549093 |  |  |  | 819200 |  |  | 6.5536e+09 | 0.01770833 | 0.001739502 | 0.0171875 | 3.052032 | 0.2226562 | 4.656592 | 7036.906 |
| BD3-LM | byte_block16_trained_100k | 100000 | 0.6825194 |  | 0.6825194 |  |  | 0.9846674 | 1.978857 | 0.5010107 | 819200 | 0 | 0.001 | 6.5536e+09 | 0.01663603 | 0.00201416 | 0.01868873 | 3.033608 | 0.2578125 | 9.172907 | 3572.259 |

## Generation benchmark

| model_family | model_variant | evaluation_label | point | nfe | n_samples | generation_seconds |
|---|---|---|---|---|---|---|
| CFM-M1 | full_sequence_best_step68001 | m1_generation_benchmark_n1 |  | 1 | 1 | 0.1044244 |
| CFM-M1 | full_sequence_best_step68001 | m1_generation_benchmark_n1000 |  | 1 | 1000 | 104.1364 |

## Every metric by run and measurement type

For time-series metrics, `latest_value` is the last valid value ordered by training/checkpoint step, event time, and source row.

| model_family | model_variant | run_id | measurement_type | metric_name | metric_unit | row_count | measurement_count | canonical_rows | duplicate_rows | min_value | max_value | mean_value | latest_value | min_training_step | max_training_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BCFM-M3 | block16_partial_step40000 | 2026-07-28_12-43-24_12345_local | checkpoint_metadata | checkpoint_epoch | boolean | 1 | 1 | 1 | 0 | 2 | 2 | 2 | 2 |  |  |
| BCFM-M3 | block16_partial_step40000 | 2026-07-28_12-43-24_12345_local | checkpoint_metadata | checkpoint_global_step | boolean | 1 | 1 | 1 | 0 | 40000 | 40000 | 40000 | 40000 |  |  |
| BCFM-M3 | block16_partial_step40000 | 2026-07-28_12-43-24_12345_local | checkpoint_metadata | checkpoint_loadable | boolean | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| BCFM-M3 | block16_partial_step40000 | 2026-07-28_12-43-24_12345_local | model_inventory | checkpoint_present | boolean | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 |  |  |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | checkpoint_metadata | checkpoint_epoch | boolean | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |  |  |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | checkpoint_metadata | checkpoint_global_step | boolean | 1 | 1 | 1 | 0 | 8000 | 8000 | 8000 | 8000 |  |  |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | checkpoint_metadata | checkpoint_loadable | boolean | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | frontier_integral | ratio | 4 | 4 | 4 | 0 | 1 | 1 | 1 | 1 | 4000 | 16000 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | gen_ppl | perplexity | 4 | 4 | 4 | 0 | 6.038054 | 14.18182 | 8.2009 | 14.18182 | 4000 | 16000 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | gen_ppl_seconds | seconds | 4 | 4 | 4 | 0 | 30.2871 | 31.71527 | 30.73075 | 30.49469 | 4000 | 16000 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | mauve | ratio | 4 | 4 | 4 | 0 | 0.004072096 | 0.004072096 | 0.004072096 | 0.004072096 | 4000 | 16000 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | mauve_seconds | seconds | 4 | 4 | 4 | 0 | 39.0089 | 44.87066 | 42.56785 | 39.0089 | 4000 | 16000 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | sampling_seconds | seconds | 4 | 4 | 4 | 0 | 0.6188755 | 0.6936096 | 0.6389683 | 0.6240188 | 4000 | 16000 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | token_entropy | nats_or_objective | 4 | 4 | 4 | 0 | 0 | 0.1012492 | 0.02699693 | 0.006738565 | 4000 | 16000 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | total_seconds | seconds | 4 | 4 | 4 | 0 | 70.71911 | 77.94885 | 74.52278 | 70.71911 | 4000 | 16000 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | validation_event | as_recorded | 4 | 4 | 4 | 0 | 1 | 4 | 2.5 | 4 | 4000 | 16000 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | learning_rate | learning_rate | 1626 | 1626 | 1626 | 0 | 1.8982e-05 | 0.001 | 0.000984886 | 0.001 | 9 | 16259 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | train_loss_epoch | as_recorded | 1 | 1 | 1 | 0 | 11.94685 | 11.94685 | 11.94685 | 11.94685 | 14570 | 14570 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | train_loss_step | as_recorded | 1626 | 1626 | 1626 | 0 | 10.33328 | 21.63741 | 11.9324 | 11.80029 | 9 | 16259 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | train_sd_loss | as_recorded | 1626 | 1626 | 1626 | 0 | 3.792811 | 10.82491 | 5.967765 | 5.836416 | 9 | 16259 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | train_vf_loss | as_recorded | 1626 | 1626 | 1626 | 0 | 5.814313 | 10.8125 | 5.964637 | 5.963875 | 9 | 16259 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | val_loss_epoch | as_recorded | 4 | 4 | 4 | 0 | 11.79195 | 11.92282 | 11.84283 | 11.92282 | 3999 | 15999 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | val_loss_step | as_recorded | 200 | 200 | 200 | 0 | 11.62359 | 12.1341 | 11.84082 | 11.83661 | 0 | 199 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | val_sd_loss | as_recorded | 200 | 200 | 200 | 0 | 5.784314 | 5.935632 | 5.851708 | 5.927744 | 0 | 199 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | val_vf_loss | as_recorded | 200 | 200 | 200 | 0 | 5.819513 | 6.210002 | 5.991127 | 5.935599 | 0 | 199 |
| ECLD | lr1e-3_failed_ablation | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e3_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | weight_decay | as_recorded | 1626 | 1626 | 1626 | 0 | 0.1 | 0.1 | 0.1 | 0.1 | 9 | 16259 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | checkpoint_metadata | checkpoint_epoch | boolean | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | checkpoint_metadata | checkpoint_global_step | boolean | 1 | 1 | 1 | 0 | 20000 | 20000 | 20000 | 20000 |  |  |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | checkpoint_metadata | checkpoint_loadable | boolean | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | external_generation_eval | frontier_integral | ratio | 9 | 9 | 9 | 0 | 0.8506087 | 0.9486856 | 0.9017345 | 0.8530334 |  |  |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | external_generation_eval | gen_ppl | perplexity | 9 | 9 | 9 | 0 | 74.13102 | 149.9478 | 113.6322 | 149.9478 |  |  |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | external_generation_eval | mauve | ratio | 9 | 9 | 9 | 0 | 0.005651912 | 0.01040985 | 0.007729003 | 0.01020356 |  |  |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | external_generation_eval | scoring_seconds | seconds | 9 | 9 | 9 | 0 | 246.3 | 288.5 | 253.0444 | 247.3 |  |  |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | frontier_integral | ratio | 25 | 25 | 25 | 0 | 0.9095495 | 0.9645896 | 0.9416954 | 0.9312522 | 4000 | 100000 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | gen_ppl | perplexity | 25 | 25 | 25 | 0 | 58.09486 | 132.5499 | 98.71937 | 116.1591 | 4000 | 100000 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | gen_ppl_seconds | seconds | 25 | 25 | 25 | 0 | 74.43684 | 126.4027 | 77.70103 | 74.62471 | 4000 | 100000 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | mauve | ratio | 25 | 25 | 25 | 0 | 0.005093743 | 0.007229547 | 0.005910395 | 0.00629458 | 4000 | 100000 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | mauve_seconds | seconds | 25 | 25 | 25 | 0 | 32.61419 | 86.32962 | 48.82142 | 49.62297 | 4000 | 100000 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | sampling_seconds | seconds | 25 | 25 | 25 | 0 | 0.6402257 | 1.397995 | 0.7079706 | 0.646357 | 4000 | 100000 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | token_entropy | nats_or_objective | 25 | 25 | 25 | 0 | 3.050646 | 3.819479 | 3.569684 | 3.763873 | 4000 | 100000 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | total_seconds | seconds | 25 | 25 | 25 | 0 | 108.809 | 215.4012 | 127.8934 | 125.5484 | 4000 | 100000 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | online_quality_curve | validation_event | as_recorded | 25 | 25 | 25 | 0 | 1 | 25 | 13 | 25 | 4000 | 100000 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | learning_rate | learning_rate | 10000 | 10000 | 10000 | 0 | 1.8982e-06 | 0.0001 | 9.975425e-05 | 0.0001 | 9 | 99999 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | train_loss_epoch | as_recorded | 7 | 7 | 7 | 0 | 9.304689 | 10.14411 | 9.488384 | 9.478009 | 14570 | 99999 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | train_loss_step | as_recorded | 10000 | 10000 | 10000 | 0 | 6.501388 | 66.77949 | 9.487475 | 8.161781 | 9 | 99999 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | train_sd_loss | as_recorded | 10000 | 10000 | 10000 | 0 | 2.574496 | 62.03771 | 4.977032 | 4.123384 | 9 | 99999 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | train_vf_loss | as_recorded | 10000 | 10000 | 10000 | 0 | 2.961939 | 10.8125 | 4.510443 | 4.038398 | 9 | 99999 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | val_loss_epoch | as_recorded | 25 | 25 | 25 | 0 | 9.036991 | 10.115 | 9.345804 | 9.384403 | 3999 | 99999 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | val_loss_step | as_recorded | 1250 | 1250 | 1250 | 0 | 6.982014 | 12.85958 | 9.343764 | 9.431772 | 0 | 1249 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | val_sd_loss | as_recorded | 1250 | 1250 | 1250 | 0 | 2.719484 | 8.353163 | 4.851714 | 4.782637 | 0 | 1249 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | val_vf_loss | as_recorded | 1250 | 1250 | 1250 | 0 | 3.076575 | 5.782147 | 4.49409 | 4.92786 | 0 | 1249 |
| ECLD | lr1e-4_fixed_reduction | 2026-08-01_tinystories_ecld_fixed_ddp2_lr1e4_warmup500_ecld1x1_384x6x6_seed12345 | training_curve | weight_decay | as_recorded | 10000 | 10000 | 10000 | 0 | 0.1 | 0.1 | 0.1 | 0.1 | 9 | 99999 |
| AR | gpt2_artifact_corrupt | AR_last_flat_artifact | checkpoint_metadata | checkpoint_loadable | boolean | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | checkpoint_metadata | checkpoint_best_metric | boolean | 1 | 1 | 1 | 0 | 0.3683956 | 0.3683956 | 0.3683956 | 0.3683956 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | checkpoint_metadata | checkpoint_ema_decay | boolean | 1 | 1 | 1 | 0 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | checkpoint_metadata | checkpoint_loadable | boolean | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | checkpoint_metadata | checkpoint_parameter_count_total | boolean | 1 | 1 | 1 | 0 | 9.259674e+07 | 9.259674e+07 | 9.259674e+07 | 9.259674e+07 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | checkpoint_metadata | checkpoint_parameter_count_trainable | boolean | 1 | 1 | 1 | 0 | 9.259674e+07 | 9.259674e+07 | 9.259674e+07 | 9.259674e+07 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | checkpoint_metadata | checkpoint_processed_tokens | tokens_or_count | 1 | 1 | 1 | 0 | 6.5536e+09 | 6.5536e+09 | 6.5536e+09 | 6.5536e+09 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | checkpoint_metadata | checkpoint_step | boolean | 1 | 1 | 1 | 0 | 100000 | 100000 | 100000 | 100000 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | checkpoint_metadata | checkpoint_wall_clock_seconds | seconds | 1 | 1 | 1 | 0 | 43547.79 | 43547.79 | 43547.79 | 43547.79 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | cross_sample_overlap_4 | ratio | 4 | 4 | 4 | 0 | 0.006162153 | 1 | 0.5227881 | 0.006162153 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | distinct_2 | ratio | 4 | 4 | 4 | 0 | 0.0009344363 | 0.4299438 | 0.1094762 | 0.4299438 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | distinct_3 | ratio | 4 | 4 | 4 | 0 | 0.001337968 | 0.6988104 | 0.1822787 | 0.6988104 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | distinct_4 | ratio | 4 | 4 | 4 | 0 | 0.001505373 | 0.8322375 | 0.2286309 | 0.8322375 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | frontier_integral | ratio | 4 | 4 | 4 | 0 | 0.4386295 | 1 | 0.720839 | 0.4447265 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | js_1gram | ratio | 4 | 4 | 4 | 0 | 0.6905966 | 0.6931472 | 0.6920025 | 0.6931472 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | js_2gram | ratio | 4 | 4 | 4 | 0 | 0.6931472 | 0.6931472 | 0.6931472 | 0.6931472 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | mauve | ratio | 4 | 4 | 4 | 0 | 0.004072096 | 0.1295738 | 0.06566708 | 0.1249503 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | scoring_seconds | seconds | 4 | 4 | 4 | 0 | 19.2 | 55.4 | 31.7 | 19.2 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | seq_rep_2 | ratio | 4 | 4 | 4 | 0 | 0.04166667 | 0.5215686 | 0.2792678 | 0.04409336 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | seq_rep_3 | ratio | 4 | 4 | 4 | 0 | 0 | 0.3149606 | 0.151213 | 0.01606925 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | seq_rep_4 | ratio | 4 | 4 | 4 | 0 | 0 | 0.229249 | 0.1025001 | 0.007116345 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | token_entropy | nats_or_objective | 4 | 4 | 4 | 0 | 2.941333 | 3.542434 | 3.258334 | 3.542434 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | external_generation_eval | zipf | as_recorded | 4 | 4 | 4 | 0 | 0.4191824 | 2.543028 | 1.314451 | 1.149525 |  |  |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | processed_training_tokens | as_recorded | 1 | 1 | 1 | 0 | 6.5536e+09 | 6.5536e+09 | 6.5536e+09 | 6.5536e+09 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | sample_adjacent_repetition_rate | ratio | 1 | 1 | 1 | 0 | 0.01770833 | 0.01770833 | 0.01770833 | 0.01770833 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | sample_distinct_1 | as_recorded | 1 | 1 | 1 | 0 | 0.001739502 | 0.001739502 | 0.001739502 | 0.001739502 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | sample_distinct_2 | as_recorded | 1 | 1 | 1 | 0 | 0.0171875 | 0.0171875 | 0.0171875 | 0.0171875 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | sample_unigram_entropy_nats | nats_or_objective | 1 | 1 | 1 | 0 | 3.052032 | 3.052032 | 3.052032 | 3.052032 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | sample_vocabulary_coverage | ratio | 1 | 1 | 1 | 0 | 0.2226562 | 0.2226562 | 0.2226562 | 0.2226562 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | sampling_seconds | seconds | 1 | 1 | 1 | 0 | 4.656592 | 4.656592 | 4.656592 | 4.656592 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | sampling_tokens_per_second | tokens/second | 1 | 1 | 1 | 0 | 7036.906 | 7036.906 | 7036.906 | 7036.906 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | test_bits_per_character | bits/character | 1 | 1 | 1 | 0 | 0.7549093 | 0.7549093 | 0.7549093 | 0.7549093 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | test_loss | as_recorded | 1 | 1 | 1 | 0 | 0.5232632 | 0.5232632 | 0.5232632 | 0.5232632 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | test_nll | nats_or_objective | 1 | 1 | 1 | 0 | 0.5232632 | 0.5232632 | 0.5232632 | 0.5232632 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | test_perplexity | perplexity | 1 | 1 | 1 | 0 | 1.687525 | 1.687525 | 1.687525 | 1.687525 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | internal_checkpoint_eval | test_token_count | tokens_or_count | 1 | 1 | 1 | 0 | 819200 | 819200 | 819200 | 819200 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_kv_cache | as_recorded | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_num_function_evaluations | as_recorded | 1 | 1 | 1 | 0 | 256 | 256 | 256 | 256 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_seed | as_recorded | 1 | 1 | 1 | 0 | 12345 | 12345 | 12345 | 12345 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_wall_clock_seconds | seconds | 1 | 1 | 1 | 0 | 4.656164 | 4.656164 | 4.656164 | 4.656164 | 100000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | train_bits_per_character | bits/character | 10001 | 10001 | 10001 | 0 | 0.6045399 | 9.118256 | 0.7152133 | 0.6283259 | 1 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | train_learning_rate | learning_rate | 10001 | 10001 | 10001 | 0 | 0 | 0.0001 | 4.999502e-05 | 0 | 1 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | train_loss | as_recorded | 10001 | 10001 | 10001 | 0 | 0.4190351 | 6.320293 | 0.4957481 | 0.4355223 | 1 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | train_nll | nats_or_objective | 10001 | 10001 | 10001 | 0 | 0.4190351 | 6.320293 | 0.4957481 | 0.4355223 | 1 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | train_processed_tokens | tokens_or_count | 10001 | 10001 | 10001 | 0 | 65536 | 6.5536e+09 | 3.2768e+09 | 6.5536e+09 | 1 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | train_token_count | tokens_or_count | 10001 | 10001 | 10001 | 0 | 65536 | 65536 | 65536 | 65536 | 1 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | val_bits_per_character | bits/character | 10 | 10 | 0 | 10 | 0.5314825 | 1.347577 | 0.6445406 | 0.5314825 | 10000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | val_loss | as_recorded | 10 | 10 | 0 | 10 | 0.3683956 | 0.9340693 | 0.4467615 | 0.3683956 | 10000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | val_nll | nats_or_objective | 10 | 10 | 0 | 10 | 0.3683956 | 0.9340693 | 0.4467615 | 0.3683956 | 10000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | tensorboard_scalar | val_token_count | tokens_or_count | 10 | 10 | 0 | 10 | 1638400 | 1638400 | 1638400 | 1638400 | 10000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | validation_curve | val_bits_per_character | bits/character | 10 | 10 | 10 | 0 | 0.5314824 | 1.347577 | 0.6445406 | 0.5314824 | 10000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | validation_curve | val_loss | as_recorded | 10 | 10 | 10 | 0 | 0.3683956 | 0.9340693 | 0.4467615 | 0.3683956 | 10000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | validation_curve | val_nll | nats_or_objective | 10 | 10 | 10 | 0 | 0.3683956 | 0.9340693 | 0.4467615 | 0.3683956 | 10000 | 100000 |
| AR | byte_trained_100k | ar_tinystories_seed12345 | validation_curve | val_token_count | tokens_or_count | 10 | 10 | 10 | 0 | 1638400 | 1638400 | 1638400 | 1638400 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | checkpoint_metadata | checkpoint_best_metric | boolean | 1 | 1 | 1 | 0 | 0.5457352 | 0.5457352 | 0.5457352 | 0.5457352 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | checkpoint_metadata | checkpoint_ema_decay | boolean | 1 | 1 | 1 | 0 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | checkpoint_metadata | checkpoint_loadable | boolean | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | checkpoint_metadata | checkpoint_parameter_count_total | boolean | 1 | 1 | 1 | 0 | 9.259674e+07 | 9.259674e+07 | 9.259674e+07 | 9.259674e+07 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | checkpoint_metadata | checkpoint_parameter_count_trainable | boolean | 1 | 1 | 1 | 0 | 9.259674e+07 | 9.259674e+07 | 9.259674e+07 | 9.259674e+07 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | checkpoint_metadata | checkpoint_processed_tokens | tokens_or_count | 1 | 1 | 1 | 0 | 6.5536e+09 | 6.5536e+09 | 6.5536e+09 | 6.5536e+09 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | checkpoint_metadata | checkpoint_step | boolean | 1 | 1 | 1 | 0 | 100000 | 100000 | 100000 | 100000 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | checkpoint_metadata | checkpoint_wall_clock_seconds | seconds | 1 | 1 | 1 | 0 | 106901.5 | 106901.5 | 106901.5 | 106901.5 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | cross_sample_overlap_4 | ratio | 26 | 26 | 26 | 0 | 0 | 0.003089328 | 0.0007805631 | 0.002837629 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | distinct_2 | ratio | 26 | 26 | 26 | 0 | 0.3609448 | 0.989002 | 0.6606988 | 0.579362 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | distinct_3 | ratio | 26 | 26 | 26 | 0 | 0.6375932 | 0.99994 | 0.8716161 | 0.8583427 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | distinct_4 | ratio | 26 | 26 | 26 | 0 | 0.8049174 | 1 | 0.9464977 | 0.948228 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | frontier_integral | ratio | 28 | 28 | 28 | 0 | 0.4662294 | 1 | 0.7439473 | 0.4830831 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | js_1gram | ratio | 26 | 26 | 26 | 0 | 0.6931472 | 0.6931472 | 0.6931472 | 0.6931472 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | js_2gram | ratio | 26 | 26 | 26 | 0 | 0.6931472 | 0.6931472 | 0.6931472 | 0.6931472 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | mauve | ratio | 28 | 28 | 28 | 0 | 0.004072096 | 0.1093303 | 0.03758672 | 0.09926685 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | scoring_seconds | seconds | 28 | 28 | 28 | 0 | 22.1 | 96.7 | 53.71786 | 75.3 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | seq_rep_2 | ratio | 26 | 26 | 26 | 0 | 0 | 0.06592899 | 0.02570518 | 0.02181597 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | seq_rep_3 | ratio | 26 | 26 | 26 | 0 | 0 | 0.02204498 | 0.005801659 | 0.003872239 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | seq_rep_4 | ratio | 26 | 26 | 26 | 0 | 0 | 0.00854177 | 0.001641149 | 0.0007231915 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | token_entropy | nats_or_objective | 28 | 28 | 28 | 0 | 0 | 3.682347 | 3.280924 | 3.585372 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | external_generation_eval | zipf | as_recorded | 26 | 26 | 26 | 0 | 0.2629596 | 1.243321 | 0.8316755 | 1.006909 |  |  |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | likelihood_is_strict_upper_bound | boolean | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | likelihood_time_min | as_recorded | 1 | 1 | 1 | 0 | 0.001 | 0.001 | 0.001 | 0.001 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | processed_training_tokens | as_recorded | 1 | 1 | 1 | 0 | 6.5536e+09 | 6.5536e+09 | 6.5536e+09 | 6.5536e+09 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | sample_adjacent_repetition_rate | ratio | 1 | 1 | 1 | 0 | 0.01663603 | 0.01663603 | 0.01663603 | 0.01663603 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | sample_distinct_1 | as_recorded | 1 | 1 | 1 | 0 | 0.00201416 | 0.00201416 | 0.00201416 | 0.00201416 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | sample_distinct_2 | as_recorded | 1 | 1 | 1 | 0 | 0.01868873 | 0.01868873 | 0.01868873 | 0.01868873 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | sample_unigram_entropy_nats | nats_or_objective | 1 | 1 | 1 | 0 | 3.033608 | 3.033608 | 3.033608 | 3.033608 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | sample_vocabulary_coverage | ratio | 1 | 1 | 1 | 0 | 0.2578125 | 0.2578125 | 0.2578125 | 0.2578125 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | sampling_seconds | seconds | 1 | 1 | 1 | 0 | 9.172907 | 9.172907 | 9.172907 | 9.172907 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | sampling_tokens_per_second | tokens/second | 1 | 1 | 1 | 0 | 3572.259 | 3572.259 | 3572.259 | 3572.259 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | test_bits_per_character_estimate | bits/character | 1 | 1 | 1 | 0 | 0.9846674 | 0.9846674 | 0.9846674 | 0.9846674 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | test_epsilon_truncated_nelbo_perplexity_estimate | perplexity | 1 | 1 | 1 | 0 | 1.978857 | 1.978857 | 1.978857 | 1.978857 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | test_loss | as_recorded | 1 | 1 | 1 | 0 | 0.6825194 | 0.6825194 | 0.6825194 | 0.6825194 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | test_masked_fraction | ratio | 1 | 1 | 1 | 0 | 0.5010107 | 0.5010107 | 0.5010107 | 0.5010107 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | test_nelbo | nats_or_objective | 1 | 1 | 1 | 0 | 0.6825194 | 0.6825194 | 0.6825194 | 0.6825194 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | internal_checkpoint_eval | test_token_count | tokens_or_count | 1 | 1 | 1 | 0 | 819200 | 819200 | 819200 | 819200 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_first_hitting_times | as_recorded | 32768 | 1 | 32768 | 0 | 4.846186e-05 | 0.999925 | 0.4993753 | 0.01667325 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_kv_cache | as_recorded | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_masked_tokens | as_recorded | 272 | 1 | 272 | 0 | 0 | 2048 | 1024 | 0 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_num_function_evaluations | as_recorded | 1 | 1 | 1 | 0 | 256 | 256 | 256 | 256 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_one_step_is_exact_final_transition | as_recorded | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_seed | as_recorded | 1 | 1 | 1 | 0 | 12345 | 12345 | 12345 | 12345 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_steps_per_block | as_recorded | 1 | 1 | 1 | 0 | 5000 | 5000 | 5000 | 5000 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_token_entropy | nats_or_objective | 256 | 1 | 256 | 0 | 0.002569016 | 2.462918 | 0.4890902 | 0.01035449 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | sampler_diagnostic | sampler_diagnostic_wall_clock_seconds | seconds | 1 | 1 | 1 | 0 | 9.17264 | 9.17264 | 9.17264 | 9.17264 | 100000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | train_bits_per_character_estimate | bits/character | 10001 | 10001 | 10001 | 0 | 0.1750256 | 1.127595 | 0.309825 | 0.2064033 | 1 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | train_learning_rate | learning_rate | 10001 | 10001 | 10001 | 0 | 0 | 0.0001 | 4.999502e-05 | 0 | 1 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | train_loss | as_recorded | 10001 | 10001 | 10001 | 0 | 0.1213185 | 0.7815896 | 0.2147543 | 0.1430678 | 1 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | train_masked_fraction | ratio | 10001 | 10001 | 10001 | 0 | 0.2760773 | 0.5049438 | 0.3030455 | 0.2793274 | 1 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | train_nelbo | nats_or_objective | 10001 | 10001 | 10001 | 0 | 0.1213185 | 0.7815896 | 0.2147543 | 0.1430678 | 1 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | train_processed_tokens | tokens_or_count | 10001 | 10001 | 10001 | 0 | 65536 | 6.5536e+09 | 3.2768e+09 | 6.5536e+09 | 1 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | train_token_count | tokens_or_count | 10001 | 10001 | 10001 | 0 | 32768 | 32768 | 32768 | 32768 | 1 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | val_bits_per_character_estimate | bits/character | 10 | 10 | 0 | 10 | 0.7873296 | 0.85253 | 0.818175 | 0.7873296 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | val_loss | as_recorded | 10 | 10 | 0 | 10 | 0.5457352 | 0.5909287 | 0.5671157 | 0.5457352 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | val_masked_fraction | ratio | 10 | 10 | 0 | 10 | 0.5000543 | 0.500993 | 0.5005244 | 0.5004431 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | val_nelbo | nats_or_objective | 10 | 10 | 0 | 10 | 0.5457352 | 0.5909287 | 0.5671157 | 0.5457352 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | val_schedule_candidates | as_recorded | 10 | 10 | 0 | 10 | 36 | 36 | 36 | 36 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | val_selected_mask_rate_max | ratio | 10 | 10 | 0 | 10 | 0.55 | 0.55 | 0.55 | 0.55 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | val_selected_mask_rate_min | ratio | 10 | 10 | 0 | 10 | 0.05 | 0.05 | 0.05 | 0.05 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | val_selected_schedule_nelbo_variance | as_recorded | 10 | 10 | 0 | 10 | 14.98389 | 21.2798 | 17.16103 | 14.98389 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | tensorboard_scalar | val_token_count | tokens_or_count | 10 | 10 | 0 | 10 | 1638400 | 1638400 | 1638400 | 1638400 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | validation_curve | val_bits_per_character_estimate | bits/character | 10 | 10 | 10 | 0 | 0.7873295 | 0.85253 | 0.818175 | 0.7873295 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | validation_curve | val_loss | as_recorded | 10 | 10 | 10 | 0 | 0.5457352 | 0.5909288 | 0.5671157 | 0.5457352 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | validation_curve | val_masked_fraction | ratio | 10 | 10 | 10 | 0 | 0.5000543 | 0.500993 | 0.5005244 | 0.5004431 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | validation_curve | val_nelbo | nats_or_objective | 10 | 10 | 10 | 0 | 0.5457352 | 0.5909288 | 0.5671157 | 0.5457352 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | validation_curve | val_schedule_candidates | as_recorded | 10 | 10 | 10 | 0 | 36 | 36 | 36 | 36 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | validation_curve | val_selected_mask_rate_max | ratio | 10 | 10 | 10 | 0 | 0.55 | 0.55 | 0.55 | 0.55 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | validation_curve | val_selected_mask_rate_min | ratio | 10 | 10 | 10 | 0 | 0.05 | 0.05 | 0.05 | 0.05 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | validation_curve | val_selected_schedule_nelbo_variance | as_recorded | 10 | 10 | 10 | 0 | 14.98389 | 21.2798 | 17.16103 | 14.98389 | 10000 | 100000 |
| BD3-LM | byte_block16_trained_100k | bd3lm_b16_tinystories_seed12345 | validation_curve | val_token_count | tokens_or_count | 10 | 10 | 10 | 0 | 1638400 | 1638400 | 1638400 | 1638400 | 10000 | 100000 |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | cross_sample_overlap_4 | ratio | 8 | 8 | 8 | 0 | 9.282178e-05 | 0.01029727 | 0.005436823 | 0.009233531 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | distinct_2 | ratio | 8 | 8 | 8 | 0 | 0.2296952 | 0.4851333 | 0.3179515 | 0.2497319 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | distinct_3 | ratio | 8 | 8 | 8 | 0 | 0.5397084 | 0.9130859 | 0.6864533 | 0.5702971 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | distinct_4 | ratio | 8 | 8 | 8 | 0 | 0.760692 | 0.9918324 | 0.8664312 | 0.7863837 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | frontier_integral | ratio | 8 | 8 | 8 | 0 | 0.04050174 | 0.9809891 | 0.3681236 | 0.04614744 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | gen_ppl | perplexity | 8 | 8 | 8 | 0 | 13.62993 | 650.0576 | 139.7493 | 17.17839 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | js_1gram | ratio | 8 | 8 | 8 | 0 | 0.01890683 | 0.04133715 | 0.02541497 | 0.02021612 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | js_2gram | ratio | 8 | 8 | 8 | 0 | 0.138761 | 0.4853205 | 0.2217502 | 0.1432133 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | mauve | ratio | 8 | 8 | 8 | 0 | 0.00458246 | 0.9507253 | 0.4966623 | 0.9360491 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | scoring_seconds | seconds | 8 | 8 | 8 | 0 | 251.9 | 1102.3 | 391.725 | 289.2 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | seq_rep_2 | ratio | 8 | 8 | 8 | 0 | 0.03572304 | 0.1625996 | 0.1109904 | 0.1554534 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | seq_rep_3 | ratio | 8 | 8 | 8 | 0 | 0.001291831 | 0.05201156 | 0.02752421 | 0.04699803 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | seq_rep_4 | ratio | 8 | 8 | 8 | 0 | 0.0001003582 | 0.02124506 | 0.009745602 | 0.01829607 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | token_entropy | nats_or_objective | 8 | 8 | 8 | 0 | 4.364337 | 4.472699 | 4.412762 | 4.369438 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | zipf | as_recorded | 8 | 8 | 8 | 0 | 1.520328 | 1.755912 | 1.567539 | 1.529885 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | sequence_latency_benchmark | sequence_latency_ms | milliseconds | 8 | 8 | 7 | 1 | 244.0762 | 4037.253 | 1401.14 | 608.8036 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | sequence_latency_benchmark | sequence_latency_p10_ms | milliseconds | 8 | 8 | 7 | 1 | 239.6572 | 3970.601 | 1380.109 | 601.6729 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | sequence_latency_benchmark | sequence_latency_p90_ms | milliseconds | 8 | 8 | 7 | 1 | 276.3495 | 4225.22 | 1481.761 | 627.1342 |  |  |
| BD3-LM | best_canonical_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | sequence_latency_benchmark | sequence_latency_repeats | count | 8 | 8 | 7 | 1 | 30 | 30 | 30 | 30 |  |  |
| BD3-LM | block16_trained_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | checkpoint_metadata | checkpoint_best_metric | boolean | 2 | 2 | 1 | 0 | 1.813493 | 1.813493 | 1.813493 | 1.813493 |  |  |
| BD3-LM | block16_trained_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | checkpoint_metadata | checkpoint_ema_decay | boolean | 2 | 2 | 1 | 0 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |  |  |
| BD3-LM | block16_trained_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | checkpoint_metadata | checkpoint_loadable | boolean | 2 | 2 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| BD3-LM | block16_trained_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | checkpoint_metadata | checkpoint_parameter_count_total | boolean | 2 | 2 | 1 | 0 | 5.111944e+07 | 5.111944e+07 | 5.111944e+07 | 5.111944e+07 |  |  |
| BD3-LM | block16_trained_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | checkpoint_metadata | checkpoint_parameter_count_trainable | boolean | 2 | 2 | 1 | 0 | 5.111944e+07 | 5.111944e+07 | 5.111944e+07 | 5.111944e+07 |  |  |
| BD3-LM | block16_trained_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | checkpoint_metadata | checkpoint_processed_tokens | tokens_or_count | 2 | 2 | 1 | 0 | 3.2768e+09 | 3.2768e+09 | 3.2768e+09 | 3.2768e+09 |  |  |
| BD3-LM | block16_trained_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | checkpoint_metadata | checkpoint_step | boolean | 2 | 2 | 1 | 0 | 100000 | 100000 | 100000 | 100000 |  |  |
| BD3-LM | block16_trained_100k | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | checkpoint_metadata | checkpoint_wall_clock_seconds | seconds | 2 | 2 | 1 | 0 | 57792.95 | 57793.76 | 57793.35 | 57793.76 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | cross_sample_overlap_4 | ratio | 8 | 8 | 0 | 8 | 9.282178e-05 | 0.01029727 | 0.005436823 | 0.009233531 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | distinct_2 | ratio | 8 | 8 | 0 | 8 | 0.2296952 | 0.4851333 | 0.3179515 | 0.2497319 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | distinct_3 | ratio | 8 | 8 | 0 | 8 | 0.5397084 | 0.9130859 | 0.6864533 | 0.5702971 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | distinct_4 | ratio | 8 | 8 | 0 | 8 | 0.760692 | 0.9918324 | 0.8664312 | 0.7863837 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | frontier_integral | ratio | 8 | 8 | 0 | 8 | 0.04050174 | 0.9809891 | 0.3681236 | 0.04614744 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | gen_ppl | perplexity | 8 | 8 | 0 | 8 | 13.62993 | 650.0576 | 139.7493 | 17.17839 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | js_1gram | ratio | 8 | 8 | 0 | 8 | 0.01890683 | 0.04133715 | 0.02541497 | 0.02021612 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | js_2gram | ratio | 8 | 8 | 0 | 8 | 0.138761 | 0.4853205 | 0.2217502 | 0.1432133 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | mauve | ratio | 8 | 8 | 0 | 8 | 0.00458246 | 0.9507253 | 0.4966623 | 0.9360491 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | scoring_seconds | seconds | 8 | 8 | 0 | 8 | 0 | 0 | 0 | 0 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | seq_rep_2 | ratio | 8 | 8 | 0 | 8 | 0.03572304 | 0.1625996 | 0.1109904 | 0.1554534 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | seq_rep_3 | ratio | 8 | 8 | 0 | 8 | 0.001291831 | 0.05201156 | 0.02752421 | 0.04699803 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | seq_rep_4 | ratio | 8 | 8 | 0 | 8 | 0.0001003582 | 0.02124506 | 0.009745602 | 0.01829607 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | token_entropy | nats_or_objective | 8 | 8 | 0 | 8 | 4.364337 | 4.472699 | 4.412762 | 4.369438 |  |  |
| BD3-LM | last_alias_bit_identical | bd3lm_tinystories_gpt2_h384_l6_b16_seed12345 | external_generation_eval | zipf | as_recorded | 8 | 8 | 0 | 8 | 1.520328 | 1.755912 | 1.567539 | 1.529885 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | cross_sample_overlap_4 | ratio | 1 | 1 | 1 | 0 | 0.01241551 | 0.01241551 | 0.01241551 | 0.01241551 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | distinct_2 | ratio | 1 | 1 | 1 | 0 | 0.1630323 | 0.1630323 | 0.1630323 | 0.1630323 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | distinct_3 | ratio | 1 | 1 | 1 | 0 | 0.4365004 | 0.4365004 | 0.4365004 | 0.4365004 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | distinct_4 | ratio | 1 | 1 | 1 | 0 | 0.6649484 | 0.6649484 | 0.6649484 | 0.6649484 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | frontier_integral | ratio | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | gen_ppl | perplexity | 1 | 1 | 1 | 0 | 6.468816 | 6.468816 | 6.468816 | 6.468816 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | js_1gram | ratio | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | js_2gram | ratio | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | mauve | ratio | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | scoring_seconds | seconds | 1 | 1 | 1 | 0 | 1095.4 | 1095.4 | 1095.4 | 1095.4 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | seq_rep_2 | ratio | 1 | 1 | 1 | 0 | 0.1709463 | 0.1709463 | 0.1709463 | 0.1709463 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | seq_rep_3 | ratio | 1 | 1 | 1 | 0 | 0.06695412 | 0.06695412 | 0.06695412 | 0.06695412 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | seq_rep_4 | ratio | 1 | 1 | 1 | 0 | 0.02936056 | 0.02936056 | 0.02936056 | 0.02936056 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | token_entropy | nats_or_objective | 1 | 1 | 1 | 0 | 4.376747 | 4.376747 | 4.376747 | 4.376747 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | external_generation_eval | zipf | as_recorded | 1 | 1 | 1 | 0 | 1.683027 | 1.683027 | 1.683027 | 1.683027 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | cross_sample_overlap_4 | ratio | 1 | 1 | 0 | 0 | 0.01241551 | 0.01241551 | 0.01241551 | 0.01241551 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | distinct_2 | ratio | 1 | 1 | 0 | 0 | 0.1630323 | 0.1630323 | 0.1630323 | 0.1630323 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | distinct_3 | ratio | 1 | 1 | 0 | 0 | 0.4365004 | 0.4365004 | 0.4365004 | 0.4365004 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | distinct_4 | ratio | 1 | 1 | 0 | 0 | 0.6649484 | 0.6649484 | 0.6649484 | 0.6649484 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | js_1gram | ratio | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | js_2gram | ratio | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | scoring_seconds | seconds | 1 | 1 | 0 | 0 | 1.7 | 1.7 | 1.7 | 1.7 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | seq_rep_2 | ratio | 1 | 1 | 0 | 0 | 0.1709463 | 0.1709463 | 0.1709463 | 0.1709463 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | seq_rep_3 | ratio | 1 | 1 | 0 | 0 | 0.06695412 | 0.06695412 | 0.06695412 | 0.06695412 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | seq_rep_4 | ratio | 1 | 1 | 0 | 0 | 0.02936056 | 0.02936056 | 0.02936056 | 0.02936056 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | token_entropy | nats_or_objective | 1 | 1 | 0 | 0 | 4.376747 | 4.376747 | 4.376747 | 4.376747 |  |  |
| gold_reference | validation_reference | gold_tinystories_n2048 | sanity_eval_duplicate | zipf | as_recorded | 1 | 1 | 0 | 0 | 1.683027 | 1.683027 | 1.683027 | 1.683027 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | cross_sample_overlap_4 | ratio | 6 | 6 | 6 | 0 | 0.005748213 | 0.01027491 | 0.008557169 | 0.01027491 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | distinct_2 | ratio | 6 | 6 | 6 | 0 | 0.1328527 | 0.2553615 | 0.2134386 | 0.2148055 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | distinct_3 | ratio | 6 | 6 | 6 | 0 | 0.3978877 | 0.5924043 | 0.5186732 | 0.515502 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | distinct_4 | ratio | 6 | 6 | 6 | 0 | 0.6525117 | 0.8131485 | 0.7484325 | 0.7427047 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | frontier_integral | ratio | 6 | 6 | 6 | 0 | 0.04312114 | 0.08981373 | 0.05518124 | 0.04373464 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | gen_ppl | perplexity | 6 | 6 | 6 | 0 | 15.40935 | 26.99332 | 18.82323 | 15.40935 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | js_1gram | ratio | 6 | 6 | 6 | 0 | 0.01667773 | 0.0241778 | 0.02246133 | 0.02405893 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | js_2gram | ratio | 6 | 6 | 6 | 0 | 0.1065631 | 0.1514783 | 0.1366147 | 0.138917 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | mauve | ratio | 6 | 6 | 6 | 0 | 0.8183863 | 0.9445849 | 0.9133172 | 0.9436481 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | scoring_seconds | seconds | 6 | 6 | 6 | 0 | 281.1 | 783.6 | 369.05 | 298.3 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | seq_rep_2 | ratio | 6 | 6 | 6 | 0 | 0.1321768 | 0.1706419 | 0.1586624 | 0.1706419 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | seq_rep_3 | ratio | 6 | 6 | 6 | 0 | 0.0332954 | 0.05491049 | 0.04789802 | 0.05491049 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | seq_rep_4 | ratio | 6 | 6 | 6 | 0 | 0.01016706 | 0.02259604 | 0.01809343 | 0.02197073 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | token_entropy | nats_or_objective | 6 | 6 | 6 | 0 | 4.332745 | 4.412504 | 4.359385 | 4.332745 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | zipf | as_recorded | 6 | 6 | 6 | 0 | 1.587269 | 1.811147 | 1.627267 | 1.59347 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | sequence_latency_benchmark | sequence_latency_ms | milliseconds | 5 | 5 | 5 | 0 | 195.4245 | 2315.566 | 1004.526 | 2315.566 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | sequence_latency_benchmark | sequence_latency_p10_ms | milliseconds | 5 | 5 | 5 | 0 | 193.1746 | 2261.444 | 983.7252 | 2261.444 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | sequence_latency_benchmark | sequence_latency_p90_ms | milliseconds | 5 | 5 | 5 | 0 | 198.4278 | 2359.78 | 1024.039 | 2359.78 |  |  |
| MDLM | best_canonical_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | sequence_latency_benchmark | sequence_latency_repeats | count | 5 | 5 | 5 | 0 | 30 | 30 | 30 | 30 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | cross_sample_overlap_4 | ratio | 6 | 6 | 0 | 6 | 0.005748213 | 0.01027491 | 0.008557169 | 0.01027491 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | distinct_2 | ratio | 6 | 6 | 0 | 6 | 0.1328527 | 0.2553615 | 0.2134386 | 0.2148055 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | distinct_3 | ratio | 6 | 6 | 0 | 6 | 0.3978877 | 0.5924043 | 0.5186732 | 0.515502 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | distinct_4 | ratio | 6 | 6 | 0 | 6 | 0.6525117 | 0.8131485 | 0.7484325 | 0.7427047 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | frontier_integral | ratio | 6 | 6 | 0 | 6 | 0.04312114 | 0.08981373 | 0.05518124 | 0.04373464 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | gen_ppl | perplexity | 6 | 6 | 0 | 6 | 15.40935 | 26.99332 | 18.82323 | 15.40935 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | js_1gram | ratio | 6 | 6 | 0 | 6 | 0.01667773 | 0.0241778 | 0.02246133 | 0.02405893 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | js_2gram | ratio | 6 | 6 | 0 | 6 | 0.1065631 | 0.1514783 | 0.1366147 | 0.138917 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | mauve | ratio | 6 | 6 | 0 | 6 | 0.8183863 | 0.9445849 | 0.9133172 | 0.9436481 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | scoring_seconds | seconds | 6 | 6 | 0 | 6 | 0 | 0 | 0 | 0 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | seq_rep_2 | ratio | 6 | 6 | 0 | 6 | 0.1321768 | 0.1706419 | 0.1586624 | 0.1706419 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | seq_rep_3 | ratio | 6 | 6 | 0 | 6 | 0.0332954 | 0.05491049 | 0.04789802 | 0.05491049 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | seq_rep_4 | ratio | 6 | 6 | 0 | 6 | 0.01016706 | 0.02259604 | 0.01809343 | 0.02197073 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | token_entropy | nats_or_objective | 6 | 6 | 0 | 6 | 4.332745 | 4.412504 | 4.359385 | 4.332745 |  |  |
| MDLM | last_alias_bit_identical | mdlm_tinystories_gpt2_h384_l6_seed12345 | external_generation_eval | zipf | as_recorded | 6 | 6 | 0 | 6 | 1.587269 | 1.811147 | 1.627267 | 1.59347 |  |  |
| MDLM | trained_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | checkpoint_metadata | checkpoint_best_metric | boolean | 2 | 2 | 1 | 0 | 1.973708 | 1.973708 | 1.973708 | 1.973708 |  |  |
| MDLM | trained_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | checkpoint_metadata | checkpoint_ema_decay | boolean | 2 | 2 | 1 | 0 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |  |  |
| MDLM | trained_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | checkpoint_metadata | checkpoint_loadable | boolean | 2 | 2 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| MDLM | trained_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | checkpoint_metadata | checkpoint_parameter_count_total | boolean | 2 | 2 | 1 | 0 | 5.111944e+07 | 5.111944e+07 | 5.111944e+07 | 5.111944e+07 |  |  |
| MDLM | trained_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | checkpoint_metadata | checkpoint_parameter_count_trainable | boolean | 2 | 2 | 1 | 0 | 5.111944e+07 | 5.111944e+07 | 5.111944e+07 | 5.111944e+07 |  |  |
| MDLM | trained_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | checkpoint_metadata | checkpoint_processed_tokens | tokens_or_count | 2 | 2 | 1 | 0 | 3.2768e+09 | 3.2768e+09 | 3.2768e+09 | 3.2768e+09 |  |  |
| MDLM | trained_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | checkpoint_metadata | checkpoint_step | boolean | 2 | 2 | 1 | 0 | 100000 | 100000 | 100000 | 100000 |  |  |
| MDLM | trained_100k | mdlm_tinystories_gpt2_h384_l6_seed12345 | checkpoint_metadata | checkpoint_wall_clock_seconds | seconds | 2 | 2 | 1 | 0 | 29842.7 | 29843.5 | 29843.1 | 29843.5 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | checkpoint_metadata | checkpoint_epoch | boolean | 1 | 1 | 1 | 0 | 4 | 4 | 4 | 4 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | checkpoint_metadata | checkpoint_global_step | boolean | 1 | 1 | 1 | 0 | 68001 | 68001 | 68001 | 68001 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | checkpoint_metadata | checkpoint_loadable | boolean | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | cross_sample_overlap_4 | ratio | 14 | 14 | 14 | 0 | 0.005134949 | 0.008594925 | 0.007176314 | 0.007640353 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | distinct_2 | ratio | 14 | 14 | 14 | 0 | 0.1153876 | 0.3524893 | 0.2409264 | 0.3468367 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | distinct_3 | ratio | 14 | 14 | 14 | 0 | 0.5230376 | 0.7051858 | 0.6287162 | 0.699065 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | distinct_4 | ratio | 14 | 14 | 14 | 0 | 0.8241493 | 0.8738266 | 0.8517632 | 0.8677356 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | frontier_integral | ratio | 14 | 14 | 14 | 0 | 0.2689904 | 0.9147611 | 0.621347 | 0.3066607 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | gen_ppl | perplexity | 14 | 14 | 14 | 0 | 74.8903 | 100.716 | 88.84245 | 94.33208 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | js_1gram | ratio | 14 | 14 | 14 | 0 | 0.03540193 | 0.1180765 | 0.06675742 | 0.04128587 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | js_2gram | ratio | 14 | 14 | 14 | 0 | 0.2095363 | 0.3266114 | 0.2484191 | 0.215215 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | mauve | ratio | 14 | 14 | 14 | 0 | 0.006974877 | 0.3440323 | 0.1100912 | 0.2802168 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | scoring_seconds | seconds | 14 | 14 | 14 | 0 | 154 | 299.8 | 210 | 162.4 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | seq_rep_2 | ratio | 14 | 14 | 14 | 0 | 0.09450061 | 0.1418199 | 0.1118966 | 0.09666054 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | seq_rep_3 | ratio | 14 | 14 | 14 | 0 | 0.02867403 | 0.03324926 | 0.03047117 | 0.02991203 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | seq_rep_4 | ratio | 14 | 14 | 14 | 0 | 0.007750741 | 0.01076149 | 0.009866535 | 0.01069201 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | token_entropy | nats_or_objective | 14 | 14 | 14 | 0 | 4.049144 | 4.554597 | 4.363306 | 4.554597 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | external_generation_eval | zipf | as_recorded | 14 | 14 | 14 | 0 | 1.398543 | 2.364817 | 1.856944 | 1.421028 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | generation_benchmark | generation_seconds | seconds | 2 | 2 | 2 | 0 | 0.1044244 | 104.1364 | 52.12041 | 104.1364 |  |  |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | epoch | as_recorded | 10106 | 10106 | 10106 | 0 | 0 | 6 | 2.940332 | 6 | 9 | 99999 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | lr_AdamW | as_recorded | 40000 | 40000 | 40000 | 0 | 1.8982e-06 | 0.0001 | 9.975424e-05 | 0.0001 | 9 | 99999 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | lr_AdamW_weight_decay | as_recorded | 40000 | 40000 | 40000 | 0 | 0.1 | 0.1 | 0.1 | 0.1 | 9 | 99999 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | train_loss_epoch | as_recorded | 7 | 7 | 7 | 0 | 4.014173 | 4.741512 | 4.164545 | 4.014173 | 14570 | 99999 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | train_loss_step | as_recorded | 10000 | 10000 | 10000 | 0 | 2.118238 | 10.8125 | 4.163534 | 3.934491 | 9 | 99999 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | train_sd_loss | as_recorded | 10000 | 10000 | 10000 | 0 | 5.541929e-11 | 0.09101959 | 0.0243242 | 0.02356397 | 9 | 99999 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | train_vf_loss | as_recorded | 10000 | 10000 | 10000 | 0 | 2.107383 | 10.8125 | 4.13921 | 3.910927 | 9 | 99999 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | val_entropy_1_steps | nats_or_objective | 99 | 99 | 99 | 0 | 0 | 4.246918 | 3.969031 | 4.195247 | 999 | 99001 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | val_loss_epoch | as_recorded | 99 | 99 | 99 | 0 | 3.915756 | 5.977965 | 4.147188 | 4.106484 | 999 | 99001 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | val_loss_step | as_recorded | 4950 | 4950 | 4950 | 0 | 2.029865 | 6.181352 | 4.147188 | 4.240022 | 0 | 4949 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | val_sd_loss | as_recorded | 4950 | 4950 | 4950 | 0 | 3.854172e-07 | 0.08586558 | 0.02328735 | 0.03300142 | 0 | 4949 |
| CFM-M1 | full_sequence_best_step68001 | tinystories_a100 | tensorboard_scalar | val_vf_loss | as_recorded | 4950 | 4950 | 4950 | 0 | 1.992227 | 6.181351 | 4.123901 | 4.20702 | 0 | 4949 |
