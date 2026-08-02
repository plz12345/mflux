from mflux.models.common.lora.mapping.lora_mapping import DiffTarget, LoRAMapping, LoRATarget

# Prefixes seen across exporters for the same underlying module path.
_PREFIXES = ("", "transformer.", "diffusion_model.", "base_model.model.")


class Krea2LoRAMapping(LoRAMapping):
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []
        targets.extend(Krea2LoRAMapping._get_global_targets())
        targets.extend(Krea2LoRAMapping._get_text_fusion_targets("layerwise_blocks"))
        targets.extend(Krea2LoRAMapping._get_text_fusion_targets("refiner_blocks"))
        targets.extend(Krea2LoRAMapping._get_transformer_block_targets())
        return targets

    @staticmethod
    def get_diff_mapping() -> list[DiffTarget]:
        # ComfyUI-format Krea 2 adapters ship the fully fine-tuned RMSNorm scales and
        # modulation vectors as `.diff` deltas, since those have no low-rank form.
        targets: list[DiffTarget] = [
            Krea2LoRAMapping._diff("txtmlp.norm.scale", aliases=["txtmlp.0.scale", "txt_in.norm.scale"]),
            Krea2LoRAMapping._diff("last.norm.scale", aliases=["final_layer.norm.scale"]),
            Krea2LoRAMapping._diff("last.modulation.lin", aliases=["final_layer.modulation.lin"]),
        ]

        mlx = "blocks.{block}"
        diffusers = "transformer_blocks.{block}"
        targets.append(Krea2LoRAMapping._diff(f"{mlx}.mod.lin", aliases=[f"{diffusers}.mod.lin"]))
        targets.extend(Krea2LoRAMapping._norm_diffs(mlx, diffusers))

        for group in ("layerwise_blocks", "refiner_blocks"):
            targets.extend(
                Krea2LoRAMapping._norm_diffs(f"txtfusion.{group}.{{block}}", f"text_fusion.{group}.{{block}}")
            )

        return targets

    @staticmethod
    def _norm_diffs(mlx: str, diffusers: str) -> list[DiffTarget]:
        suffixes = ("prenorm.scale", "postnorm.scale", "attn.qknorm.qnorm.scale", "attn.qknorm.knorm.scale")
        return [Krea2LoRAMapping._diff(f"{mlx}.{s}", aliases=[f"{diffusers}.{s}"]) for s in suffixes]

    @staticmethod
    def _diff(param_path: str, aliases: list[str] | None = None) -> DiffTarget:
        return DiffTarget(
            param_path=param_path,
            possible_patterns=[
                f"{prefix}{path}.diff" for path in (param_path, *(aliases or [])) for prefix in _PREFIXES
            ],
        )

    @staticmethod
    def _get_global_targets() -> list[LoRATarget]:
        return [
            Krea2LoRAMapping._target("first", aliases=["img_in"]),
            Krea2LoRAMapping._target("tmlp.linear_in", aliases=["tmlp.0", "time_embed.linear_1"]),
            Krea2LoRAMapping._target("tmlp.linear_out", aliases=["tmlp.2", "time_embed.linear_2"]),
            Krea2LoRAMapping._target("tproj.linear", aliases=["tproj.1", "time_mod_proj"]),
            Krea2LoRAMapping._target("txtmlp.linear_in", aliases=["txtmlp.1", "txt_in.linear_1"]),
            Krea2LoRAMapping._target("txtmlp.linear_out", aliases=["txtmlp.3", "txt_in.linear_2"]),
            Krea2LoRAMapping._target("txtfusion.projector", aliases=["text_fusion.projector"]),
            Krea2LoRAMapping._target("last.linear", aliases=["final_layer.linear"]),
        ]

    @staticmethod
    def _get_text_fusion_targets(group: str) -> list[LoRATarget]:
        # MLX modules use the official names; diffusers exports use text_fusion.*/ff.*/to_*.
        mlx = f"txtfusion.{group}.{{block}}"
        diffusers = f"text_fusion.{group}.{{block}}"
        return [
            Krea2LoRAMapping._target(f"{mlx}.attn.wq", aliases=[f"{diffusers}.attn.to_q"]),
            Krea2LoRAMapping._target(f"{mlx}.attn.wk", aliases=[f"{diffusers}.attn.to_k"]),
            Krea2LoRAMapping._target(f"{mlx}.attn.wv", aliases=[f"{diffusers}.attn.to_v"]),
            Krea2LoRAMapping._target(f"{mlx}.attn.gate", aliases=[f"{diffusers}.attn.to_gate"]),
            Krea2LoRAMapping._target(
                f"{mlx}.attn.wo",
                aliases=[f"{diffusers}.attn.to_out.0", f"{diffusers}.attn.to_out"],
            ),
            Krea2LoRAMapping._target(f"{mlx}.mlp.gate", aliases=[f"{diffusers}.ff.gate"]),
            Krea2LoRAMapping._target(f"{mlx}.mlp.up", aliases=[f"{diffusers}.ff.up"]),
            Krea2LoRAMapping._target(f"{mlx}.mlp.down", aliases=[f"{diffusers}.ff.down"]),
        ]

    @staticmethod
    def _get_transformer_block_targets() -> list[LoRATarget]:
        mlx = "blocks.{block}"
        diffusers = "transformer_blocks.{block}"
        return [
            Krea2LoRAMapping._target(f"{mlx}.attn.wq", aliases=[f"{diffusers}.attn.to_q"]),
            Krea2LoRAMapping._target(f"{mlx}.attn.wk", aliases=[f"{diffusers}.attn.to_k"]),
            Krea2LoRAMapping._target(f"{mlx}.attn.wv", aliases=[f"{diffusers}.attn.to_v"]),
            Krea2LoRAMapping._target(f"{mlx}.attn.gate", aliases=[f"{diffusers}.attn.to_gate"]),
            Krea2LoRAMapping._target(
                f"{mlx}.attn.wo",
                aliases=[f"{diffusers}.attn.to_out.0", f"{diffusers}.attn.to_out"],
            ),
            Krea2LoRAMapping._target(f"{mlx}.mlp.gate", aliases=[f"{diffusers}.ff.gate"]),
            Krea2LoRAMapping._target(f"{mlx}.mlp.up", aliases=[f"{diffusers}.ff.up"]),
            Krea2LoRAMapping._target(f"{mlx}.mlp.down", aliases=[f"{diffusers}.ff.down"]),
        ]

    @staticmethod
    def _target(model_path: str, aliases: list[str] | None = None) -> LoRATarget:
        module_paths = [model_path, *(aliases or [])]
        flat_paths = [Krea2LoRAMapping._flatten(path) for path in module_paths]
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=Krea2LoRAMapping._matrix_patterns(module_paths, flat_paths, "up"),
            possible_down_patterns=Krea2LoRAMapping._matrix_patterns(module_paths, flat_paths, "down"),
            possible_alpha_patterns=Krea2LoRAMapping._alpha_patterns(module_paths, flat_paths),
            possible_lokr_w1_patterns=Krea2LoRAMapping._lokr_patterns(module_paths, "lokr_w1"),
            possible_lokr_w2_patterns=Krea2LoRAMapping._lokr_patterns(module_paths, "lokr_w2"),
        )

    @staticmethod
    def _matrix_patterns(module_paths: list[str], flat_paths: list[str], direction: str) -> list[str]:
        if direction == "up":
            suffixes = [
                "lora_B.weight",
                "lora_B.default.weight",
                # ComfyUI-format exports store the matrix at the bare key, without `.weight`
                "lora_B",
                "lora_up.weight",
                "lora_up.default.weight",
                "lora_up",
                "lora.up.weight",
                "lora.up.default.weight",
            ]
        else:
            suffixes = [
                "lora_A.weight",
                "lora_A.default.weight",
                "lora_A",
                "lora_down.weight",
                "lora_down.default.weight",
                "lora_down",
                "lora.down.weight",
                "lora.down.default.weight",
            ]

        patterns = []
        for path in module_paths:
            for prefix in _PREFIXES:
                patterns.extend(f"{prefix}{path}.{suffix}" for suffix in suffixes)

        flat_suffix = "lora_up" if direction == "up" else "lora_down"
        for path in flat_paths:
            patterns.extend(
                [
                    f"lora_unet_{path}.{flat_suffix}.weight",
                    f"lora_unet_{path}.{flat_suffix}.default.weight",
                ]
            )

        return patterns

    @staticmethod
    def _lokr_patterns(module_paths: list[str], factor: str) -> list[str]:
        # ai-toolkit exports LoKr factors as `<prefix><path>.lokr_w1` / `.lokr_w2`
        # (full matrices) using the official Krea module names. The loader derives
        # the decomposed `_a`/`_b`/`lokr_t2` variants from these base patterns.
        patterns = []
        for path in module_paths:
            patterns.extend(f"{prefix}{path}.{factor}" for prefix in _PREFIXES)
        return patterns

    @staticmethod
    def _alpha_patterns(module_paths: list[str], flat_paths: list[str]) -> list[str]:
        patterns = []
        for path in module_paths:
            patterns.extend(f"{prefix}{path}.alpha" for prefix in _PREFIXES)
        patterns.extend(f"lora_unet_{path}.alpha" for path in flat_paths)
        return patterns

    @staticmethod
    def _flatten(path: str) -> str:
        return path.replace(".", "_")
