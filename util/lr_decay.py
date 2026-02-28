def get_layer_id_for_vit_str(name: str, num_layers: int) -> int:
    if name.startswith("encoder.cls_token") or name.startswith("encoder.pos_embed"):
        return 0
    if name.startswith("encoder.patch_embed"):
        return 0
    if name.startswith("encoder.blocks"):
        return int(name.split(".")[2]) + 1
    return num_layers + 1


def param_groups_lrd(model, weight_decay=0.05, layer_decay=0.75, no_weight_decay_list=()):
    """Layer-wise LR decay for ViT-encoder + recognition head.

    Encoder input layers get smaller lr_scale, upper layers / recognition head get larger lr_scale.
    """

    param_group_names = {}
    param_groups = {}

    num_layers = len(model.encoder.blocks)
    total_groups = num_layers + 2
    layer_scales = [layer_decay ** (total_groups - i) for i in range(total_groups + 1)]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 1 or name in no_weight_decay_list:
            group_decay = "no_decay"
            this_decay = 0.0
        else:
            group_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit_str(name, num_layers)
        group_name = f"layer_{layer_id}_{group_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(name)
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())
