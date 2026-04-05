import os
import numpy as np
import keras

# ── Compatibility patch ───────────────────────────────────────────────────────
# Models saved in Colab (Keras 3.x) include 'quantization_config' in Dense/Conv2D
# configs. Some local Keras builds don't accept it in from_config(). Patch it out.
def _strip_quant(layer_cls):
    original = layer_cls.from_config.__func__

    @classmethod
    def _patched(cls, config):
        config = dict(config)
        config.pop('quantization_config', None)
        return original(cls, config)

    layer_cls.from_config = _patched

for _cls in (keras.layers.Dense, keras.layers.Conv2D,
             keras.layers.DepthwiseConv2D, keras.layers.Embedding):
    _strip_quant(_cls)
# ─────────────────────────────────────────────────────────────────────────────


def _load(path):
    return keras.models.load_model(path)


class HierarchicalMoE:
    def __init__(self, models_dir='saved_models'):
        self.models_dir = models_dir

        print("Loading experts...")
        self.base_expert = _load(os.path.join(models_dir, 'base_expert_final.keras'))
        self.art_expert  = _load(os.path.join(models_dir, 'artificial_expert_final.keras'))
        self.nat_expert  = _load(os.path.join(models_dir, 'natural_expert_final.keras'))

        print("Loading gating networks...")
        self.first_level_gate = _load(os.path.join(models_dir, 'first_level_gate_final.keras'))
        self.art_gater = _load(os.path.join(models_dir, 'artificial_gater_final.keras'))
        self.nat_gater = _load(os.path.join(models_dir, 'natural_gater_final.keras'))

        print("All models loaded successfully.")

    def predict(self, x, batch_size=32, verbose=1):
        p_nat = self.first_level_gate.predict(x, batch_size=batch_size, verbose=verbose)
        p_art = 1.0 - p_nat

        w_art = self.art_gater.predict(x, batch_size=batch_size, verbose=verbose)
        w_nat = self.nat_gater.predict(x, batch_size=batch_size, verbose=verbose)

        pred_base = self.base_expert.predict(x, batch_size=batch_size, verbose=verbose)
        pred_art  = self.art_expert.predict(x, batch_size=batch_size, verbose=verbose)
        pred_nat  = self.nat_expert.predict(x, batch_size=batch_size, verbose=verbose)

        branch_art = w_art[:, 0:1] * pred_base + w_art[:, 1:2] * pred_art
        branch_nat = w_nat[:, 0:1] * pred_base + w_nat[:, 1:2] * pred_nat

        return p_art * branch_art + p_nat * branch_nat

    def evaluate(self, x, y_true_oh, batch_size=32):
        preds = self.predict(x, batch_size=batch_size, verbose=0)
        acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_true_oh, axis=1))
        return float(acc)

    def get_routing_info(self, x):
        p_nat = self.first_level_gate.predict(x, verbose=0)
        w_art = self.art_gater.predict(x, verbose=0)
        w_nat = self.nat_gater.predict(x, verbose=0)
        return {
            'prob_natural':      p_nat,
            'prob_artificial':   1.0 - p_nat,
            'weight_art_branch': w_art,
            'weight_nat_branch': w_nat
        }
