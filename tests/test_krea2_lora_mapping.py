import mlx.core as mx
import pytest

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping

pytestmark = pytest.mark.fast


def _tiny_transformer() -> Krea2Transformer:
    # head_dim = features // heads = 16 keeps the RoPE axis split valid; tiny everywhere else.
    return Krea2Transformer(
        features=32,
        tdim=16,
        txtdim=16,
        heads=2,
        kvheads=1,
        multiplier=2,
        layers=1,
        patch=2,
        channels=16,
        theta=1000,
        txtlayers=12,
        txtheads=2,
        txtkvheads=1,
    )


class TestKrea2LoRAMapping:
    def test_matches_official_krea_keys(self):
        keys = [
            "blocks.0.attn.wq.lora_A.weight",
            "blocks.0.attn.wq.lora_B.weight",
            "blocks.0.attn.wo.lora_down.weight",
            "blocks.0.attn.wo.lora_up.weight",
            "blocks.0.mlp.gate.lora_A.weight",
            "blocks.0.mlp.gate.lora_B.weight",
            "txtfusion.layerwise_blocks.0.attn.wq.lora_A.weight",
            "txtfusion.layerwise_blocks.0.attn.wq.lora_B.weight",
            "txtfusion.refiner_blocks.1.mlp.down.lora_A.weight",
            "txtfusion.refiner_blocks.1.mlp.down.lora_B.weight",
            "first.lora_A.weight",
            "first.lora_B.weight",
            "tmlp.0.lora_A.weight",
            "tmlp.0.lora_B.weight",
            "tproj.1.lora_A.weight",
            "tproj.1.lora_B.weight",
            "last.linear.lora_A.weight",
            "last.linear.lora_B.weight",
        ]
        assert self._matched_keys(keys) == set(keys)

    def test_matches_diffusers_and_flat_keys(self):
        keys = [
            "transformer.transformer_blocks.0.attn.to_q.lora_A.default.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_B.default.weight",
            "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_B.weight",
            "transformer.transformer_blocks.0.ff.gate.lora.down.weight",
            "transformer.transformer_blocks.0.ff.gate.lora.up.weight",
            "base_model.model.transformer_blocks.0.attn.to_gate.lora_down.weight",
            "base_model.model.transformer_blocks.0.attn.to_gate.lora_up.weight",
            "text_fusion.refiner_blocks.1.ff.down.lora_A.weight",
            "text_fusion.refiner_blocks.1.ff.down.lora_B.weight",
            "lora_unet_blocks_0_attn_wq.lora_down.weight",
            "lora_unet_blocks_0_attn_wq.lora_up.weight",
        ]
        assert self._matched_keys(keys) == set(keys)

    def test_matches_official_krea_collection_export_keys(self):
        # krea/Krea-2-LoRA-* uses transformer.* diffusers exports (e.g. retroanime).
        keys = [
            "transformer.img_in.lora_A.weight",
            "transformer.img_in.lora_B.weight",
            "transformer.final_layer.linear.lora_A.weight",
            "transformer.final_layer.linear.lora_B.weight",
            "transformer.text_fusion.layerwise_blocks.0.attn.to_gate.lora_A.weight",
            "transformer.text_fusion.layerwise_blocks.0.attn.to_gate.lora_B.weight",
            "transformer.text_fusion.projector.lora_A.weight",
            "transformer.text_fusion.projector.lora_B.weight",
        ]
        assert self._matched_keys(keys) == set(keys)

    def test_matches_peft_base_model_export_keys(self):
        # gokaygokay/Krea-2-Realism-LoRA and similar trainers use base_model.model.* with official names.
        keys = [
            "base_model.model.blocks.0.attn.wq.lora_A.weight",
            "base_model.model.blocks.0.attn.wq.lora_B.weight",
            "base_model.model.blocks.0.attn.gate.lora_A.weight",
            "base_model.model.blocks.0.attn.gate.lora_B.weight",
            "base_model.model.txtfusion.refiner_blocks.1.mlp.down.lora_A.weight",
            "base_model.model.txtfusion.refiner_blocks.1.mlp.down.lora_B.weight",
        ]
        assert self._matched_keys(keys) == set(keys)

    def test_applies_official_block_lora_to_transformer(self, tmp_path):
        transformer = _tiny_transformer()
        lora_path = tmp_path / "krea2_lora.safetensors"
        mx.save_safetensors(
            str(lora_path),
            {
                "blocks.0.attn.wq.lora_A.weight": mx.ones((2, 32)),
                "blocks.0.attn.wq.lora_B.weight": mx.ones((32, 2)),
            },
        )

        lora_paths, lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=transformer,
            lora_paths=[str(lora_path)],
            lora_scales=[0.7],
            bake_lora=False,
        )

        target = transformer.blocks[0].attn.wq
        assert lora_paths == [str(lora_path)]
        assert lora_scales == [pytest.approx(0.7)]
        assert isinstance(target, LoRALinear)
        assert target.scale == pytest.approx(0.7)

    def test_applies_diffusers_block_lora_to_transformer(self, tmp_path):
        transformer = _tiny_transformer()
        lora_path = tmp_path / "krea2_diffusers_lora.safetensors"
        mx.save_safetensors(
            str(lora_path),
            {
                "transformer.transformer_blocks.0.ff.down.lora_A.weight": mx.ones((2, 128)),
                "transformer.transformer_blocks.0.ff.down.lora_B.weight": mx.ones((32, 2)),
            },
        )

        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=transformer,
            lora_paths=[str(lora_path)],
            lora_scales=[1.0],
            bake_lora=False,
        )

        assert isinstance(transformer.blocks[0].mlp.down, LoRALinear)

    def test_applies_krea_collection_projector_lora_to_transformer(self, tmp_path):
        transformer = _tiny_transformer()
        lora_path = tmp_path / "krea2_projector_lora.safetensors"
        mx.save_safetensors(
            str(lora_path),
            {
                "transformer.text_fusion.projector.lora_A.weight": mx.ones((2, 16)),
                "transformer.text_fusion.projector.lora_B.weight": mx.ones((16, 2)),
            },
        )

        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=transformer,
            lora_paths=[str(lora_path)],
            lora_scales=[0.5],
            bake_lora=False,
        )

        assert isinstance(transformer.txtfusion.projector, LoRALinear)
        assert transformer.txtfusion.projector.scale == pytest.approx(0.5)

    def test_matches_comfy_bare_matrix_keys(self):
        # ComfyUI-format exports (e.g. lodestones/Kroma) drop the trailing `.weight`.
        keys = [
            "diffusion_model.blocks.0.attn.wq.lora_A",
            "diffusion_model.blocks.0.attn.wq.lora_B",
            "diffusion_model.blocks.0.mlp.down.lora_down",
            "diffusion_model.blocks.0.mlp.down.lora_up",
            "diffusion_model.txtfusion.projector.lora_A",
            "diffusion_model.txtfusion.projector.lora_B",
        ]
        assert self._matched_keys(keys) == set(keys)

    def test_applies_comfy_bare_matrix_lora_to_transformer(self, tmp_path):
        transformer = _tiny_transformer()
        lora_path = tmp_path / "krea2_comfy_lora.safetensors"
        mx.save_safetensors(
            str(lora_path),
            {
                "diffusion_model.blocks.0.attn.wq.lora_A": mx.ones((2, 32)),
                "diffusion_model.blocks.0.attn.wq.lora_B": mx.ones((32, 2)),
            },
        )

        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=transformer,
            lora_paths=[str(lora_path)],
            lora_scales=[0.7],
            bake_lora=False,
        )

        assert isinstance(transformer.blocks[0].attn.wq, LoRALinear)

    def test_applies_weight_diffs_to_transformer(self, tmp_path):
        transformer = _tiny_transformer()
        prenorm_before = transformer.blocks[0].prenorm.scale
        mod_before = transformer.blocks[0].mod.lin
        txtmlp_before = transformer.txtmlp.norm.scale

        prenorm_delta = mx.ones_like(prenorm_before)
        mod_delta = mx.ones_like(mod_before)
        txtmlp_delta = mx.ones_like(txtmlp_before)

        lora_path = tmp_path / "krea2_diff_lora.safetensors"
        mx.save_safetensors(
            str(lora_path),
            {
                # A low-rank pair so the file is a valid adapter on its own.
                "diffusion_model.blocks.0.attn.wq.lora_A": mx.ones((2, 32)),
                "diffusion_model.blocks.0.attn.wq.lora_B": mx.ones((32, 2)),
                "diffusion_model.blocks.0.prenorm.scale.diff": prenorm_delta,
                "diffusion_model.blocks.0.mod.lin.diff": mod_delta,
                "diffusion_model.txtmlp.0.scale.diff": txtmlp_delta,
            },
        )

        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            diff_mapping=Krea2LoRAMapping.get_diff_mapping(),
            transformer=transformer,
            lora_paths=[str(lora_path)],
            lora_scales=[0.5],
            bake_lora=False,
        )

        # `.diff` deltas are added in place, attenuated by the LoRA scale.
        assert mx.allclose(transformer.blocks[0].prenorm.scale, prenorm_before + 0.5 * prenorm_delta)
        assert mx.allclose(transformer.blocks[0].mod.lin, mod_before + 0.5 * mod_delta)
        assert mx.allclose(transformer.txtmlp.norm.scale, txtmlp_before + 0.5 * txtmlp_delta)

    def test_applies_diff_only_adapter(self, tmp_path):
        transformer = _tiny_transformer()
        before = transformer.last.norm.scale
        delta = mx.ones_like(before)

        lora_path = tmp_path / "krea2_diff_only.safetensors"
        mx.save_safetensors(str(lora_path), {"diffusion_model.last.norm.scale.diff": delta})

        # An adapter carrying only deltas must not trip the "nothing was applied" guard.
        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            diff_mapping=Krea2LoRAMapping.get_diff_mapping(),
            transformer=transformer,
            lora_paths=[str(lora_path)],
            lora_scales=[1.0],
            bake_lora=False,
        )

        assert mx.allclose(transformer.last.norm.scale, before + delta)

    def test_diff_mapping_ignored_when_not_supplied(self, tmp_path):
        transformer = _tiny_transformer()
        lora_path = tmp_path / "krea2_diff_unsupported.safetensors"
        mx.save_safetensors(str(lora_path), {"diffusion_model.last.norm.scale.diff": mx.ones((32,))})

        with pytest.raises(ValueError, match="No LoRA layers were applied"):
            LoRALoader.load_and_apply_lora(
                lora_mapping=Krea2LoRAMapping.get_mapping(),
                transformer=transformer,
                lora_paths=[str(lora_path)],
                lora_scales=[1.0],
                bake_lora=False,
            )

    def _matched_keys(self, keys: list[str]) -> set[str]:
        matched_keys: set[str] = set()
        for key in keys:
            for target in Krea2LoRAMapping.get_mapping():
                pattern_groups = (
                    target.possible_up_patterns,
                    target.possible_down_patterns,
                    target.possible_alpha_patterns,
                )
                for patterns in pattern_groups:
                    if any(LoRALoader._match_pattern(key, pattern) is not None for pattern in patterns):
                        matched_keys.add(key)
                        break
                if key in matched_keys:
                    break
        return matched_keys

    def test_latent_creator_noise_shape(self):
        from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator

        noise = Krea2LatentCreator.create_noise(seed=0, height=1024, width=1024)
        assert noise.shape == (1, 16, 128, 128)
