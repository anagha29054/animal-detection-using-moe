"""
Run this to debug model loading:
    python debug_load.py
"""
import traceback
import keras

print(f"Keras version: {keras.__version__}")

# Apply the same patch as moe_model.py
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

print("Patch applied. Attempting to load base_expert_final.keras ...")
try:
    m = keras.models.load_model('saved_models/base_expert_final.keras')
    print("SUCCESS:", m.summary())
except Exception as e:
    print("\n--- LOAD FAILED ---")
    traceback.print_exc()
    print("\nFailing layer config excerpt:", str(e)[:500])
