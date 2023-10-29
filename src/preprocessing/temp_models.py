import timm


efficientnet_models = [
    'efficientnet_b0', 'efficientnet_b0_g8_gn', 'efficientnet_b0_g16_evos', 'efficientnet_b0_gn',
    'efficientnet_b1', 'efficientnet_b1_pruned', 'efficientnet_b2', 'efficientnet_b2_pruned', 'efficientnet_b2a',
    'efficientnet_b3', 'efficientnet_b3_g8_gn', 'efficientnet_b3_gn', 'efficientnet_b3_pruned', 'efficientnet_b3a',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8',
    'efficientnet_cc_b0_4e', 'efficientnet_cc_b0_8e', 'efficientnet_cc_b1_8e', 'efficientnet_el',
    'efficientnet_el_pruned', 'efficientnet_em', 'efficientnet_es', 'efficientnet_es_pruned', 'efficientnet_l2',
    'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2', 'efficientnet_lite3', 'efficientnet_lite4',
    'efficientnetv2_l', 'efficientnetv2_m', 'efficientnetv2_rw_m', 'efficientnetv2_rw_s', 'efficientnetv2_rw_t',
    'efficientnetv2_s', 'efficientnetv2_xl'
]
vit_models_256 = [
    'vit_base_patch16_reg8_gap_256',
    'vit_base_patch16_siglip_256',
    'vit_base_patch32_plus_256',
    'vit_large_patch16_siglip_256',
    'vit_medium_patch16_gap_256',
    'vit_medium_patch16_reg4_256',
    'vit_medium_patch16_reg4_gap_256',
    'vit_relpos_base_patch32_plus_rpn_256'
]
# Load each model and count parameters
param_counts = {}
for model_name in efficientnet_models:
    model = timm.create_model(model_name, pretrained=False)
    param_counts[model_name] = sum(p.numel() for p in model.parameters())

# Sort models by parameter count
sorted_models = sorted(param_counts.items(), key=lambda x: x[1])
i = 0
list_eff = []
# Print models sorted by parameter count
for model_name, param_count in sorted_models:
    print(f"{model_name}: {param_count} parameters")
    i += 1
    list_eff.append(model_name)
    if i == 5:
        break

# Load each model and count parameters
param_counts = {}
for model_name in vit_models_256:
    model = timm.create_model(model_name, pretrained=False)
    param_counts[model_name] = sum(p.numel() for p in model.parameters())

# Sort models by parameter count
sorted_models = sorted(param_counts.items(), key=lambda x: x[1])


i = 0
list_vit = []
# Print models sorted by parameter count
for model_name, param_count in sorted_models:
    i += 1
    print(f"{model_name}: {param_count} parameters")
    list_vit.append(model_name)
    if i == 3:
        break
pass